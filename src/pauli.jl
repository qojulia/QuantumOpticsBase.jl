
# comp stands for computational basis
function _comp_pauli_1(basis_fn)
    b = SpinBasis(1//2)
    vec_it(fn) = vec(fn(b).data)/2 # provides "standard" normalization
    V = hcat(map(vec_it, [identityoperator, sigmax, sigmay, sigmaz])...)
    Operator(basis_fn(b, b), PauliBasis(1), V)
end

_comp_pauli_kb1_cached = _comp_pauli_1(KetBraBasis)
_comp_pauli_choi1_cached = _comp_pauli_1(ChoiBasis)

# TODO: should this be further cached?
"""
    pauli_comp_kb(N)

Creates a superoperator which changes from the computational `KetBra(SpinBasis(1//2))`
to the `PauliBasis()` over `N` qubits.
"""
comp_pauli_kb(N::Integer) = tensor_pow(_comp_pauli_kb1_cached, N)
comp_pauli_choi(N::Integer) = tensor_pow(_comp_pauli_choi1_cached, N)

function _check_is_spinbasis(b)
    for i=1:length(b)
        (b[i] isa SpinBasis && dimension(b[i]) == 2) || throw(ArgumentError("Superoperator must be over systems composed of SpinBasis(1//2) to be converted to pauli representation"))
    end
end

# It's possible to get better asympotic speedups using, e.g. methods from
# https://iopscience.iop.org/article/10.1088/1402-4896/ad6499
# https://arxiv.org/abs/2411.00526
# https://quantum-journal.org/papers/q-2024-09-05-1461/ (see appendices)
function pauli(op::Operator)
    (basis_l(op) == basis_r(op)) || throw(ArgumentError("Operator must be square in order to be vectorized to the pauli represenation"))
    _check_is_spinbasis(basis_l(op))
    dagger(comp_pauli_kb(length(basis_l(op)))) * vec(op)
end

function pauli(op::SOpKetBraType)
    bl, br = basis_l(op), basis_r(op)
    ((basis_l(bl) == basis_r(bl)) && (basis_l(br) == basis_r(br))) || throw(ArgumentError("Superoperator must map between square operators in order to be converted to pauli represenation"))
    bl, br = basis_l(bl), basis_l(br)
    foreach(_check_is_spinbasis, (bl, br))

    Nl, Nr = length(bl), length(br)
    Vl = comp_pauli_kb(Nl)
    Vr = Nl == Nr ? Vl : comp_pauli_kb(Nr)
    #data = dagger(Ul)*op.data*Ur # TODO figure out normalization
    #@assert isapprox(imag.(data), zero(data), atol=tol)
    #Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
    dagger(Vl) * op * Vr # TODO figure out normalization
    # TODO make sure dagger here is okay...
end

function chi(op::ChoiStateType)
    (basis_l(op) == basis_r(op)) || throw(ArgumentError("Choi state must map between square operators in order to be converted to chi represenation"))
    bl, br = basis_l(basis_l(op)), basis_r(basis_l(op))
    foreach(_check_is_spinbasis, (bl, br))

    Nl, Nr = length(bl), length(br)
    Vl = comp_pauli_kb(Nl)
    Vr = Nl == Nr ? Vl : comp_pauli_kb(Nr)
    dagger(Vl) * op * Vr # TODO figure out normalization
end

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
