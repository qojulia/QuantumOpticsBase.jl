import QuantumInterface: KetBraBasis, ChoiBasis, PauliBasis
using TensorCast

const SOpBasis = Union{KetBraBasis, PauliBasis}
const SuperOperatorType{BL,BR,T} = Operator{BL,BR,T} where {BL<:SOpBasis,BR<:SOpBasis}
const SOpKetBraType{BL,BR,T} = Operator{BL,BR,T} where {BL<:KetBraBasis,BR<:KetBraBasis}
const SOpPauliType{BL,BR,T} = Operator{BL,BR,T} where {BL<:PauliBasis,BR<:PauliBasis}
const ChoiStateType{BL,BR,T} = Operator{BL,BR,T} where {BR<:ChoiBasis,BL<:ChoiBasis}
const ChoiKetBraType{BL,BR,T} = Operator{ChoiBasis{BL,BR},ChoiBasis{BL,BR},T} where {BL<:KetBraBasis,BR<:KetBraBasis}
const ChiType{BL,BR,T} = Operator{ChoiBasis{BL,BR},ChoiBasis{BL,BR},T} where {BL<:PauliBasis,BR<:PauliBasis}

const DenseSuperOpPureType{BL,BR} = SuperOperatorType{BL,BR,<:Matrix}
const DenseSuperOpAdjType{BL,BR} = SuperOperatorType{BL,BR,<:Adjoint{<:Number,<:Matrix}}
const DenseSuperOpType{BL,BR} = Union{DenseOpPureType{BL,BL},DenseOpAdjType{BL,BR}}
const SparseSuperOpPureType{BL,BR} = SuperOperatorType{BL,BR,<:SparseMatrixCSC}
const SparseSuperOpAdjType{BL,BR} = SuperOperatorType{BL,BR,<:Adjoint{<:Number,<:SparseMatrixCSC}}
const SparseSuperOpType{BL,BR} = Union{SparseOpPureType{BL,BR},SparseOpAdjType{BL,BR}}

vec(op::Operator) = Ket(KetBraBasis(basis_l(op), basis_r(op)), vec(op.data))
#function unvec(k::Ket{<:KetBraBasis})
#    bl, br = basis_l(basis(k)), basis_r(basis(k))
#    Operator(bl, br, reshape(k.data, dimension(bl), dimension(br)))
#end
function unvec(k::Ket{<:KetBraBasis})
    bl, br = basis_l(basis(k)), basis_r(basis(k))
    @cast A[n,m] |= k.data[(n,m)] (n ∈ 1:dimension(bl), m ∈ 1:dimension(br))
    return Operator(bl, br, A)
end

function spre(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    sprepost(op, identityoperator(op))
end

function spost(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spost of a non-square operator should be. See issue #113"))
    sprepost(identityoperator(op), op)
end

#sprepost(A::Operator, B::Operator) = Operator(KetBraBasis(basis_l(A), basis_r(B)), KetBraBasis(basis_r(A), basis_l(B)), kron(permutedims(B.data), A.data))
function sprepost(A::Operator, B::Operator)
    @cast C[(ν,μ), (n,m)] |= A.data[ν,n] * B.data[m,μ]
    Operator(KetBraBasis(basis_l(A), basis_r(B)), KetBraBasis(basis_r(A), basis_l(B)), C)
end

function tensor(A::SuperOperatorType, B::SuperOperatorType)
    a1, a2 = basis_l(A), basis_r(A)
    b1, b2 = basis_l(B), basis_r(B)
    da1, da2, db1, db2 = map(dimension, (a1, a2, b1, b2))
    @cast A[(ν,μ), (n,m)] |= A[m,μ] * B[n,ν] (m ∈ 1:da1, μ ∈ 1:da2, n ∈ 1:db1, ν ∈ 1:db2)
    return Operator(a1⊗b2, a2⊗b2, A)
end

#function _sch(data, l1, l2, r1, r2)
#    data = reshape(data, map(dimension, (l1, l2, r1, r2)))
#    data = permutedims(data, (4, 2, 3, 1))
#    reshape(data, map(dimension, (l2⊗r2, l1⊗r1)))
#end
#
#function _sch(data::SparseMatrixCSC, l1, l2, r1, r2)
#    #data = reshape(sparse(data), map(dimension, (l1, l2, r1, r2)), ())
#    #data = _permutedims(sparse(data), size(data), (4, 2, 3, 1))
#    data = _permutedims(sparse(data), map(dimension, (l1, l2, r1, r2)), (4, 2, 3, 1))
#    sparse(reshape(data, map(dimension, (l2⊗r2, l1⊗r1))))
#    # sparse(::ReshapedArray) only works when ndims == 2
#end

#sprepost(A::Operator, B::Operator) = Operator(KetBraBasis(basis_l(A), basis_r(B)), KetBraBasis(basis_r(A), basis_l(B)), kron(permutedims(B.data), A.data))

#function _super_choi((l1, l2), (r1, r2), data)
#    data = Base.ReshapedArray(data, map(length, (l2, l1, r2, r1)), ())
#    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
#    data = PermutedDimsArray(data, (1, 3, 2, 4))
#    data = Base.ReshapedArray(data, map(length, (l1⊗l2, r1⊗r2)), ())
#    return (l1, l2), (r1, r2), copy(data)
#end

# Sec IV.A. of https://arxiv.org/abs/1111.6950
function _super_choi(basis_fn, op)
    l1, l2 = basis_l(basis_l(op)), basis_r(basis_l(op))
    r1, r2 = basis_l(basis_r(op)), basis_r(basis_r(op))
#    data = _sch(op.data, l1, l2, r1, r2)
#    data = reshape(op.data, map(dimension, (l1, l2, r1, r2)))
#    data = permutedims(data, (4, 2, 3, 1))
#    data = reshape(data, map(dimension, (l2⊗r2, l1⊗r1)))
    dl1, dl2, dr1, dr2 = map(dimension, (l1, l2, r1, r2))
    @cast A[(ν,μ), (n,m)] |= op.data[(m,μ), (n,ν)] (m ∈ 1:dl1, μ ∈ 1:dl2, n ∈ 1:dr1, ν ∈ 1:dr2)
    return Operator(basis_fn(r2, l2), basis_fn(r1, l1), A)
end

choi(op::SuperOperatorType) = _super_choi(ChoiBasis, op)
super(op::ChoiStateType) =  _super_choi(KetBraBasis, op)

# I'm not sure this is actually right... see sec V.C. of https://arxiv.org/abs/1111.6950
dagger(a::ChoiStateType) = choi(dagger(super(a)))

# This method is necessary so we don't fall back to the method below it
*(a::SuperOperatorType, b::SuperOperatorType) = (check_multiplicable(a,b); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SuperOperatorType, b::Operator) = unvec(a*vec(b))
*(a::ChoiStateType, b::SuperOperatorType) = super(a)*b
*(a::SuperOperatorType, b::ChoiStateType) = a*super(b)
*(a::ChoiStateType, b::ChoiStateType) = choi(super(a)*super(b))
*(a::ChoiStateType, b::Operator) = super(a)*b

identitysuperoperator(b::Basis) = Operator(KetBraBasis(b,b), KetBraBasis(b,b), Eye{ComplexF64}(dimension(b)^2))

identitysuperoperator(op::DenseSuperOpType) = 
    Operator(op.basis_l, op.basis_r, Matrix(one(eltype(op.data))I, size(op.data)))

identitysuperoperator(op::SparseSuperOpType) = 
    Operator(op.basis_l, op.basis_r, sparse(one(eltype(op.data))I, size(op.data)))

