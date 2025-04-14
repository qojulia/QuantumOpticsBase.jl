
# comp stands for computational basis
function _pauli_comp_1(b_out_fn, b_in)
    b = SpinBasis(1//2)
    vec_it(fn) = vec((fn(b)/sqrt(2)).data) # provides "standard" normalization
    V = hcat(map(vec_it, [identityoperator, sigmax, sigmay, sigmaz])...)
    Operator(b_out_fn(b, b), b_in, V)
end

_pauli_comp_kb1_cached = _pauli_comp_1(KetBraBasis, PauliBasis(1))
_pauli_comp_choi1_cached = _pauli_comp_1(ChoiBasis, ChiBasis(2))

# TODO: should this be further cached?
"""
    pauli_comp_kb(N)

Creates a superoperator which changes from the computational `KetBra(SpinBasis(1//2))`
to the `PauliBasis()` over `N` qubits.
"""
pauli_comp_kb(N::Integer) = tensor_pow(_pauli_comp_kb1_cached, N)
pauli_comp_choi(N::Integer) = tensor_pow(_pauli_comp_choi1_cached, N)

# It's possible to get better asympotic speedups using, e.g. methods from
# https://iopscience.iop.org/article/10.1088/1402-4896/ad6499
# https://arxiv.org/abs/2411.00526
# https://quantum-journal.org/papers/q-2024-09-05-1461/ (see appendices)
pauli(op::Operator) = dagger(pauli_comp_kb(length(basis(op)))) * vec(op)

function pauli(op::SOpKetBraType)
    Nl, Nr = length(basis_l(basis_l(op))), length(basis_l(basis_r(op)))
    Vl = pauli_comp_kb(Nl)
    Vr = Nl == Nr ? Vl : pauli_comp_kb(Nr)
    dagger(Vl) * op * Vr
end

function chi(op::ChoiStateType)
    Nl, Nr = length(basis_l(basis_l(op))), length(basis_r(basis_l(op)))
    Vl = pauli_comp_choi(Nl)
    Vr = Nl == Nr ? Vl : pauli_comp_choi(Nr)
    dagger(Vl) * op * Vr
end

function _pauli_chi(basis_fn, op)
    bl, br = basis_l(op), basis_r(op)
    Nl, Nr = length(bl), length(br)
    data = reshape(op.data, (2^Nl, 2^Nl, 2^Nr, 2^Nr))
    data = PermutedDimsArray(data, (4, 2, 3, 1))
    data = reshape(data, (2^(Nl+Nr), 2^(Nl+Nr)))
    return Operator(basis_fn(Nl+Nr), basis_fn(Nl+Nr), data)
end

pauli(op::ChiType) = _pauli_chi(PauliBasis, op)
chi(op::SOpPauliType) = _pauli_chi(ChiBasis, op)
pauli(op::ChoiStateType) = pauli(chi(op))
chi(op::SOpKetBraType) = chi(pauli(op))

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
