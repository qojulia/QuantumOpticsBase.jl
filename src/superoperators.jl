import QuantumInterface: KetBraBasis, ChoiBasis

const SuperOperatorType = Operator{<:KetBraBasis,<:KetBraBasis}
const ChoiStateType = Operator{CompositeBasis{<:Integer,<:ChoiBasis},CompositeBasis{<:Integer,<:ChoiBasis}}

vec(op::Operator) = Ket(KetBraBasis(basis_l(op), basis_r(op)), reshape(op.data, length(op.data)))

function spre(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    Operator(KetBraBasis(basis_l(op)), KetBraBasis(basis_r(op)), tensor(op, identityoperator(op)).data)
end

function spost(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spost of a non-square operator should be. See issue #113"))
    SuperOperator(KetBraBasis(basis_r(op)), (op.basis_l, op.basis_l), kron(permutedims(op.data), identityoperator(op).data))
end

sprepost(A::Operator, B::Operator) = SuperOperator((A.basis_l, B.basis_r), (A.basis_r, B.basis_l), kron(permutedims(B.data), A.data))

# reshape swaps within systems due to colum major ordering
# https://docs.qojulia.org/quantumobjects/operators/#tensor_order
function _super_choi((l1, l2), (r1, r2), data)
    data = reshape(data, map(length, (l2, l1, r2, r1)))
    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    data = permutedims(data, (1, 3, 2, 4))
    data = reshape(data, map(length, (l1⊗l2, r1⊗r2)))
    return (l1, l2), (r1, r2), data
end

function _super_choi((r2, l2), (r1, l1), data::SparseMatrixCSC)
    data = _permutedims(data, map(length, (l2, r2, l1, r1)), (1, 3, 2, 4))
    data = reshape(data, map(length, (l1⊗l2, r1⊗r2)))
    # sparse(data) is necessary since reshape of a sparse array returns a
    # ReshapedSparseArray which is not a subtype of AbstractArray and so
    # _permutedims fails to acces the ".m" field
    # https://github.com/qojulia/QuantumOpticsBase.jl/pull/83
    # https://github.com/JuliaSparse/SparseArrays.jl/issues/24
    # permutedims in SparseArrays.jl only implements perm (2,1) and so
    # _permutedims should probably be upstreamed
    # https://github.com/JuliaLang/julia/issues/26534
    return (l1, l2), (r1, r2), sparse(data)
end

function choi(op::SuperOperatorType)
    bl, br = basis_l(op), basis_r(op)
    (l1, l2), (r1, r2) = (basis_l(bl), basis_r(bl)), (basis_l(br), basis_r(br))
    (l1, l2), (r1, r2), data = _super_choi((l1, l2), (r1, r2), op.data)
    return Operator(ChoiBasis(l1,true)⊗ChoiBasis(l2,false),
                    ChoiBasis(r1,true)⊗ChoiBasis(r2,false), data)
end

function super(op::ChoiStateType)
    bl, br = basis_l(op), basis_r(op)
    (l1, l2), (r1, r2) = (basis_l(bl), basis_r(bl)), (basis_l(br), basis_r(br))
    (l1, l2), (r1, r2), data = _super_choi((l1, l2), (r1, r2), op.data)
    return Operator(KetBraBasis(l1,l2), KetBraBasis(r1,r2), data)
end

dagger(a::ChoiStateType) = choi(dagger(super(a)))

*(a::SuperOperatorType, b::Operator) = a*vec(b)
*(a::ChoiStateType, b::SuperOperatorType) = super(a)*b
*(a::SuperOperatorType, b::ChoiStateType) = a*super(b)
*(a::ChoiStateType, b::ChoiStateType) = choi(super(a)*super(b))
*(a::ChoiStateType, b::Operator) = super(a)*b
