import QuantumInterface: PauliBasis, ChiBasis

const PauliTransferType{BL,BR,T} = Operator{BL,BR,T} where {BL<:PauliBasis,BR<:PauliBasis}
const ChiType{BL,BR,T} = Operator{BL,BR,T} where {BL<:ChiBasis,BR<:ChiBasis}

# comp stands for computational basis
function _pauli_comp_1()
    b = SpinBasis(1//2)
    vec_it(fn) = vec((fn(b)/sqrt(2)).data) # provides "standard" normalization
    V = hcat(map(vec_it, [identityoperator, sigmax, sigmay, sigmaz])...)
    Operator(KetBraBasis(b, b), PauliBasis(1), V)
end

tensor(A::Operator{BL,BR}, B::Operator{BL,BR}) where {BL<:KetBraBasis, BR<:PauliBasis} = super_tensor(A,B)

_pauli_comp_1_cached = _pauli_comp_1()

# TODO: should this be further cached?
"""
    pauli_comp(N)

Creates a superoperator which changes from the computational `KetBra(SpinBasis(1//2))`
to the `PauliBasis()` over `N` qubits.
"""
pauli_comp(N) = tensor_pow(_pauli_comp_1_cached, N)

"""
    pauli_comp(Nl,Nr)

Creates a superoperator which changes from a Choi state in the computational `SpinBasis(1//2)` with `Nl` in the reference system and `Nr` in the output system to a `ChiBasis(Nl,Nr)`.
"""
function choi_chi(Nl, Nr)
    @assert (Nl+Nr)%2 == 0
    b = SpinBasis(1//2)
    Operator(ChoiBasis(b^Nl, b^Nr), ChiBasis(Nl, Nr), pauli_comp((Nl+Nr)÷2).data)
end

# It's possible to get better asympotic speedups using, e.g. methods from
# https://quantum-journal.org/papers/q-2024-09-05-1461/ (see appendices)
# https://iopscience.iop.org/article/10.1088/1402-4896/ad6499
# https://arxiv.org/abs/2411.00526
# https://arxiv.org/abs/2408.06206
# https://quantumcomputing.stackexchange.com/questions/31788/how-to-write-the-iswap-unitary-as-a-linear-combination-of-tensor-products-betw/31790#31790
# So probably using https://github.com/JuliaMath/Hadamard.jl would be best
function _pauli_comp_convert(op, rev)
    Nl, Nr = length(basis_l(basis_l(op))), length(basis_l(basis_r(op)))
    Vl = pauli_comp(Nl)
    Vr = Nl == Nr ? Vl : pauli_comp(Nr)
    rev ? Vl * op * dagger(Vr) : dagger(Vl) * op * Vr
end

function _choi_chi_convert(op, rev)
    Nl, Nr = length(basis_l(basis_l(op))), length(basis_r(basis_l(op)))
    V = choi_chi(Nl,Nr)
    norm = 2^((Nl+Nr)÷2)
    out = rev ? norm * V * op * dagger(V) : (1/norm) * dagger(V) * op * V
end

pauli(op::SuperOperatorType) = _pauli_comp_convert(op, false)
super(op::PauliTransferType) = _pauli_comp_convert(op, true)
chi(op::ChoiStateType) = _choi_chi_convert(op, false)
choi(op::ChiType) = _choi_chi_convert(op, true)

super(op::ChiType) = super(choi(op))
choi(op::PauliTransferType) = choi(super(op))
pauli(op::ChoiStateType) = pauli(super(op))
chi(op::SuperOperatorType) = chi(choi(op))
pauli(op::ChiType) = pauli(super(choi(op)))
chi(op::PauliTransferType) = chi(choi(super(op)))

#pauli(op::ChiType) = _super_choi(PauliBasis, op)
#chi(op::PauliTransferType) = _super_choi(ChiBasis, op)

pauli(op::PauliTransferType) = op
chi(op::ChiType) = op

pauli(op::Operator) = pauli(vec(op))
pauli(k::Ket{<:KetBraBasis}) = dagger(pauli_comp(length(basis_l(basis(k))))) * k
unvec(k::Ket{<:PauliBasis}) = unvec(pauli_comp(length(basis_l(basis(k)))) * k)

dagger(a::ChiType) = chi(dagger(choi(a)))

# TODO: document return types of mixed superoperator multiplication...
# This method is necessary so we don't fall back to the method below it
*(a::PauliTransferType, b::PauliTransferType) = (check_multiplicable(a,b); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::PauliTransferType, b::Operator) = unvec(a*pauli(b))

*(a::ChiType, b::ChiType) = chi(pauli(a)*pauli(b))
*(a::ChiType, b::Operator) = pauli(a)*b

*(a::PauliTransferType, b::SuperOperatorType) = super(a)*b
*(a::SuperOperatorType, b::PauliTransferType) = a*super(b)

*(a::PauliTransferType, b::ChoiStateType) = a*chi(b)
*(a::ChoiStateType, b::PauliTransferType) = chi(a)*b

*(a::SuperOperatorType, b::ChiType) = a*super(b)
*(a::ChiType, b::SuperOperatorType) = super(a)*b

*(a::PauliTransferType, b::ChiType) = a*pauli(b)
*(a::ChiType, b::PauliTransferType) = pauli(a)*b

*(a::ChoiStateType, b::ChiType) = a*choi(b)
*(a::ChiType, b::ChoiStateType) = choi(a)*b

"""
function hwpauli(op::SuperOperatorType; tol=1e-9)
    bl, br = basis_l(op), basis_r(op)
    ((basis_l(bl) == basis_r(bl)) && (basis_l(br) == basis_r(br))) || throw(ArgumentError("Superoperator must map between square operators in order to be converted to pauli represenation"))

    for b in (basis_l(bl), basis_l(br))
        for i=1:length(b)
            (b[i] isa NLevelBasis) || throw(ArgumentError("Superoperator must be defined only systems composed of NLevelBasis to be converted to pauli representation"))
        end
    end
end
"""
