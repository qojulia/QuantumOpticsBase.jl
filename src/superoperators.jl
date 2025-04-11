import QuantumInterface: KetBraBasis, ChoiBasis

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

vec(op::Operator) = Ket(KetBraBasis(basis_l(op), basis_r(op)), reshape(op.data, dimension(op.data)))
function unvec(k::SuperKetType)
    bl, br = basis_l(basis(k)), basis_r(basis(k))
    return Operator(bl, br, reshape(k.data, dimension(bl), dimension(br)))
end

function spre(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    Operator(KetBraBasis(basis_l(op)), KetBraBasis(basis_r(op)), tensor(op, identityoperator(op)).data)
end

function spost(op::Operator)
    multiplicable(op, op) || throw(ArgumentError("It's not clear what spost of a non-square operator should be. See issue #113"))
    Operator(KetBraBasis(basis_r(op)), KetBraBasis(basis_l(op)), kron(permutedims(op.data), identityoperator(op).data))
end

sprepost(A::Operator, B::Operator) = Operator(KetBraBasis(A.basis_l, B.basis_r), KetBraBasis(A.basis_r, B.basis_l), kron(permutedims(B.data), A.data))

# reshape swaps within systems due to colum major ordering
# https://docs.qojulia.org/quantumobjects/operators/#tensor_order
function _super_choi((l1, l2), (r1, r2), data)
    data = Base.ReshapedArray(data, map(length, (l2, l1, r2, r1)), ())
    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    data = PermutedDimsArray(data, (1, 3, 2, 4))
    data = Base.ReshapedArray(data, map(length, (l1⊗l2, r1⊗r2)), ())
    return (l1, l2), (r1, r2), copy(data)
end

function choi(op::SuperOperatorType)
    bl, br = basis_l(op), basis_r(op)
    (l1, l2), (r1, r2) = (basis_l(bl), basis_r(bl)), (basis_l(br), basis_r(br))
    (l1, l2), (r1, r2), data = _super_choi((l1, l2), (r1, r2), op.data)
    return Operator(ChoiBasis(l1,l2), ChoiBasis(r1,r2), data)
end

function super(op::ChoiStateType)
    bl, br = basis_l(op), basis_r(op)
    (l1, l2), (r1, r2) = (basis_l(bl), basis_r(bl)), (basis_l(br), basis_r(br))
    (l1, l2), (r1, r2), data = _super_choi((l1, l2), (r1, r2), op.data)
    return Operator(KetBraBasis(l1,l2), KetBraBasis(r1,r2), data)
end

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

function pauli_to_ket_bra(N)
    b = SpinBasis(1//2)
    paulis = Iterators.repeated([identityoperator(b), sigmax(b), sigmaz(b), sigmay(b)], N)
    reduce(hcat, [vec(tensor(p)) for p in Iterators.product(paulis...)])
end

function pauli(op::SuperOperatorType; tol=1e-9)
    bl, br = basis_l(op), basis_r(op)
    ((basis_l(bl) == basis_r(bl)) && (basis_l(br) == basis_r(br))) || throw(ArgumentError("Superoperator must map between square operators in order to be converted to pauli represenation"))

    for b in (basis_l(bl), basis_l(br))
        for i=1:length(b)
            (b[i] isa SpinBasis && dimension(b[i]) == 2) || throw(ArgumentError("Superoperator must be over systems composed of SpinBasis(1//2) to be converted to pauli representation"))
        end
    end

    Nl, Nr = length(basis_l(bl)), length(basis_l(br))
    Ul = ket_bra_to_pauli(Nl)
    Ur = Nl == Nr ? Ul : ket_bra_to_pauli(Nr)
    data = dagger(Ul)*op.data*Ur # TODO figure out normalization
    @assert isapprox(imag.(data), zero(data), atol=tol)
    Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
end

function chi(op::ChoiStateType; tol=1e-9)
    (basis_l(op) == basis_r(op)) || throw(ArgumentError("Choi state must map between square operators in order to be converted to chi represenation"))

    bl, br = basis_l(basis_l(op)), basis_r(basis_l(op))
    for b in (bl, br)
        for i=1:length(b)
            (b[i] isa NLevelBasis) || throw(ArgumentError("Choi state must be over systems composed of SpinBasis(1//2) to be converted to chi representation"))
        end
    end

    Nl, Nr = length(bl), length(br)
    Ul = ket_bra_to_pauli(Nl)
    Ur = Nl == Nr ? Ul : ket_bra_to_pauli(Nr)
    data = dagger(Ul)*op.data*Ur # TODO figure out normalization
    @assert isapprox(imag.(data), zero(data), atol=tol)
    Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
end

"""
# TODO figure out case for generalized paulis
function pauli_to_ketbra(b)
    #paulix(b[i],i)*pauliz(b[i],j) for i=1:N,j=1:N
end

function pauli(op::SuperOperatorType; tol=1e-9)
    bl, br = basis_l(op), basis_r(op)
    ((basis_l(bl) == basis_r(bl)) && (basis_l(br) == basis_r(br))) || throw(ArgumentError("Superoperator must map between square operators in order to be converted to pauli represenation"))

    for b in (basis_l(bl), basis_l(br))
        for i=1:length(b)
            (b[i] isa NLevelBasis) || throw(ArgumentError("Superoperator must be defined only systems composed of NLevelBasis to be converted to pauli representation"))
        end
    end
end
