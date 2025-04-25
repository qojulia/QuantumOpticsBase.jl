import QuantumInterface: KetBraBasis, ChoiBasis

const SuperOperatorType{BL,BR,T} = Operator{BL,BR,T} where {BL<:KetBraBasis,BR<:KetBraBasis}
const ChoiStateType{BL,BR,T} = Operator{BL,BR,T} where {BR<:ChoiBasis,BL<:ChoiBasis}

#const ChannelType = Union{SuperOperatorType, ChoiStateType, PauliTransferType, ChiType}

vec(op::Operator) = Ket(KetBraBasis(basis_l(op), basis_r(op)), vec(op.data))
function unvec(k::Ket{<:KetBraBasis})
    bl, br = basis_l(basis(k)), basis_r(basis(k))
    Operator(bl, br, reshape(k.data, dimension(bl), dimension(br)))
    #@cast A[n,m] |= k.data[(n,m)] (n ∈ 1:dimension(bl), m ∈ 1:dimension(br))
end
#function unvec(k::Ket{<:KetBraBasis})
#    bl, br = basis_l(basis(k)), basis_r(basis(k))
#    return Operator(bl, br, A)
#end

function spre(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    sprepost(op, identityoperator(op))
end

function spost(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spost of a non-square operator should be. See issue #113"))
    sprepost(identityoperator(op), op)
end

sprepost(A::Operator, B::Operator) = Operator(KetBraBasis(basis_l(A), basis_r(B)), KetBraBasis(basis_r(A), basis_l(B)), kron(permutedims(B.data), A.data))
#@cast C[(ν,μ), (n,m)] |= A.data[ν,n] * B.data[m,μ]

function super_tensor(A, B)
    all, alr = basis_l(basis_l(A)), basis_r(basis_l(A))
    arl, arr = basis_l(basis_r(A)), basis_r(basis_r(A))
    bll, blr = basis_l(basis_l(B)), basis_r(basis_l(B))
    brl, brr = basis_l(basis_r(B)), basis_r(basis_r(B))
    data = kron(B.data, A.data)
    data = reshape(data, map(dimension, (all, bll, alr, blr, arl, brl, arr, brr)))
    data = PermutedDimsArray(data, (1, 3, 2, 4, 5, 6, 7, 8))
    data = reshape(data, map(dimension, (basis_l(A)⊗basis_l(B), basis_r(A)⊗basis_r(B))))
    return Operator(basis_l(A)⊗basis_l(B), basis_r(A)⊗basis_r(B), data)
end

# copy at end is necessary to not fall back on generic gemm routines later
# https://discourse.julialang.org/t/permuteddimsarray-slower-than-permutedims/46401
# which suggetst usig something like Tullio might speed things up further
# Sec IV.A. of https://arxiv.org/abs/1111.6950
function _super_choi(basis_fn, op)
    l1, l2 = basis_l(basis_l(op)), basis_r(basis_l(op))
    r1, r2 = basis_l(basis_r(op)), basis_r(basis_r(op))
    #dl1, dl2, dr1, dr2 = map(dimension, (l1, l2, r1, r2))
    #@cast A[(ν,μ), (n,m)] |= op.data[(m,μ), (n,ν)] (m ∈ 1:dl1, μ ∈ 1:dl2, n ∈ 1:dr1, ν ∈ 1:dr2)

    #data = reshape(op.data, map(dimension, (l1, l2, r1, r2)))
    #data = PermutedDimsArray(data, (4, 2, 3, 1))
    #data = reshape(data, map(dimension, (r2⊗l2, r1⊗l1)))
    #return Operator(basis_fn(r2, l2), basis_fn(r1, l1), data)
    #data = Base.ReshapedArray(data, map(length, (l2, l1, r2, r1)), ())
    #(l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    #data = PermutedDimsArray(data, (1, 3, 2, 4))
    #data = Base.ReshapedArray(data, map(length, (l1⊗l2, r1⊗r2)), ())
    #return (l1, l2), (r1, r2), copy(data)

    data = reshape(op.data, map(dimension, (l1, l2, r1, r2)))
    data = PermutedDimsArray(data, (1, 3, 2, 4))
    data = reshape(data, map(dimension, (l1⊗r1, l2⊗r2)))
    return Operator(basis_fn(l1, r1), basis_fn(l2, r2), data)
end

choi(op::SuperOperatorType) = _super_choi(ChoiBasis, op)
super(op::ChoiStateType) = _super_choi(KetBraBasis, op)
super(op::SuperOperatorType) = op
choi(op::ChoiStateType) = op

# I'm not sure this is actually right... see sec V.C. of https://arxiv.org/abs/1111.6950
dagger(a::ChoiStateType) = choi(dagger(super(a)))

# This method is necessary so we don't fall back to the method below it
*(a::SuperOperatorType, b::SuperOperatorType) = (check_multiplicable(a,b); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SuperOperatorType, b::Operator) = unvec(a*vec(b))

*(a::ChoiStateType, b::ChoiStateType) = choi(super(a)*super(b))
*(a::ChoiStateType, b::Operator) = super(a)*b

*(a::ChoiStateType, b::SuperOperatorType) = super(a)*b
*(a::SuperOperatorType, b::ChoiStateType) = a*super(b)

identitysuperoperator(b::Basis) = Operator(KetBraBasis(b,b), KetBraBasis(b,b), Eye{ComplexF64}(dimension(b)^2))

identitysuperoperator(op::DenseOpType{BL,BR}) where {BL<:KetBraBasis, BR<:KetBraBasis} = 
    Operator(op.basis_l, op.basis_r, Matrix(one(eltype(op.data))I, size(op.data)))

identitysuperoperator(op::SparseOpType{BL,BR}) where {BL<:KetBraBasis, BR<:KetBraBasis} =
    Operator(op.basis_l, op.basis_r, sparse(one(eltype(op.data))I, size(op.data)))

