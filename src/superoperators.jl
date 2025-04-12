import QuantumInterface: KetBraBasis, ChoiBasis
using TensorCast

const SuperKetType = Ket{<:KetBraBasis}

const DenseSuperOpPureType{BL,BR} = Operator{<:KetBraBasis,<:KetBraBasis,<:Matrix}
const DenseSuperOpAdjType{BL,BR} = Operator{<:KetBraBasis,<:KetBraBasis,<:Adjoint{<:Number,<:Matrix}}
const DenseSuperOpType{BL,BR} = Union{DenseOpPureType{<:KetBraBasis,<:KetBraBasis},DenseOpAdjType{<:KetBraBasis,BR}}
const SparseSuperOpPureType{BL,BR} = Operator{<:KetBraBasis,<:KetBraBasis,<:SparseMatrixCSC}
const SparseSuperOpAdjType{BL,BR} = Operator{<:KetBraBasis,<:KetBraBasis,<:Adjoint{<:Number,<:SparseMatrixCSC}}
const SparseSuperOpType{BL,BR} = Union{SparseOpPureType{<:KetBraBasis,<:KetBraBasis},SparseOpAdjType{<:KetBraBasis,BR}}

const SuperOperatorType = Operator{<:KetBraBasis,<:KetBraBasis}
const ChoiStateType = Operator{<:ChoiBasis,<:ChoiBasis}

#const ChoiBasisType = Union{CompositeBasis{ChoiBasis,T} where {T}, CompositeBasis{ChoiBasis{B},T} where {B,T}}
#const ChoiStateType = Operator{ChoiBasisType,ChoiBasisType}
#const ChoiStateType = Union{Operator{CompositeBasis{ChoiBasis,S},CompositeBasis{ChoiBasis,T}} where {S,T}, Operator{CompositeBasis{ChoiBasis{BL},S},CompositeBasis{ChoiBasis{BR},T}} where {BL,BR,S,T}}

vec(op::Operator) = Ket(KetBraBasis(basis_l(op), basis_r(op)), vec(op.data))
function unvec(k::SuperKetType)
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

#sprepost(A::Operator, B::Operator) = Operator(KetBraBasis(A.basis_l, B.basis_r), KetBraBasis(A.basis_r, B.basis_l), kron(permutedims(B.data), A.data))

function sprepost(A::Operator, B::Operator)
    @cast C[(ν,μ), (n,m)] |= A.data[ν,n] * B.data[m,μ]
    Operator(KetBraBasis(basis_l(A), basis_r(B)), KetBraBasis(basis_r(A), basis_l(B)), C)
end

function _super_choi((l1, l2), (r1, r2), data)
    data = Base.ReshapedArray(data, map(length, (l2, l1, r2, r1)), ())
    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    data = PermutedDimsArray(data, (1, 3, 2, 4))
    data = Base.ReshapedArray(data, map(length, (l1⊗l2, r1⊗r2)), ())
    return (l1, l2), (r1, r2), copy(data)
end

# Sec IV.A. of  https://arxiv.org/abs/1111.6950
function _super_choi(basis_fn, op)
    l1, l2 = basis_l(basis_l(op)), basis_r(basis_l(op))
    r1, r2 = basis_l(basis_r(op)), basis_r(basis_r(op))
    d1, d2, d3, d4 = map(dimension, (l1, l2, r1, r2))
    @cast A[(ν,μ), (n,m)] |= op.data[(m,μ), (n,ν)] (m ∈ 1:d1, μ ∈ 1:d2, n ∈ 1:d3, ν ∈ 1:d4)
    return Operator(basis_fn(r2, l2), basis_fn(r1, l1), A)
end

choi(op::SuperOperatorType) = _super_choi(ChoiBasis, op)
super(op::ChoiStateType) =  _super_choi(KetBraBasis, op)

dagger(a::ChoiStateType) = choi(dagger(super(a)))

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

