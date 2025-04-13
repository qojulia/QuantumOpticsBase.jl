using TransmuteDims

#function _pauli_to_ketbra1()
#    b = SpinBasis(1//2)
#    vec_it(fn) = vec(fn(b).data)
#    p2kb = hcat(map(vec_it, [identityoperator, sigmax, sigmay, sigmaz])...)
#end
#
#function _ketbra_to_pauli(N)
#    T = kron((_pauli_to_ketbra1 for _=1:N)...)
#    T = reshape(T, Tuple(4 for _=1:2N))
#    T = PermutedDimsArray(T, ((2i-1 for i=1:N)..., (2i for i=1:N)...))
#    reshape(T, (4^N, 4^N))
#end
#
#function _op_to_pauli(Nl, Nr, data)
#    Ul = ket_bra_to_pauli(Nl)
#    Ur = Nl == Nr ? Ul : _ketbra_to_pauli(Nr)
#    data = dagger(Ul) * data * Ur # TODO figure out normalization
#    @assert isapprox(imag.(data), zero(data), atol=tol)
#    Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
#end

function _pauli_to_comp1()
    b = SpinBasis(1//2)
    vec_it(fn) = vec(fn(b).data)
    V = hcat(map(vec_it, [identityoperator, sigmax, sigmay, sigmaz])...)
    Operator(PauliBasis(), KetBraBasis(b, b), V)
end

_pauli_to_comp1_cached = _pauli_to_comp1()

# TODO should this be further cached?
"""
    comp_to_pauli(N)

Creates a superoperator which changes from the computational `SpinBasis(1//2)`
to the `PauliBasis()` over `N` qubits.
"""
function pauli_to_comp(N::Integer)
    N > 0 || throw(ArgumentError())
    N == 1 && return _pauli_to_comp1_cached
    V = pauli_to_comp(N÷2)
    (N%2 == 0) && return V⊗V
    return V⊗V⊗pauli_to_comp(1)
end

function _check_is_spinbasis(b)
    for i=1:length(b)
        (b[i] isa SpinBasis && dimension(b[i]) == 2) || throw(ArgumentError("Superoperator must be over systems composed of SpinBasis(1//2) to be converted to pauli representation"))
    end
end

function pauli(op::SOpKetBraType)
    ((basis_l(basis_l(op)) == basis_r(basis_l(op))) && (basis_l(br) == basis_r(basis_r(op)))) || throw(ArgumentError("Superoperator must map between square operators in order to be converted to pauli represenation"))
    bl, br = basis_l(basis_l(op)), basis_l(basis_r(op))
    foreach(_check_is_spinbasis, (bl, br))

    Nl, Nr = length(basis_l(bl)), length(basis_l(br))
    Vl = pauli_to_comp(Nl)
    Vr = Nl == Nr ? Vl : pauli_to_comp(Nr)
    #data = dagger(Ul)*op.data*Ur # TODO figure out normalization
    #@assert isapprox(imag.(data), zero(data), atol=tol)
    #Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
    dagger(Vl) * op * Vr # TODO figure out normalization
end

function chi(op::ChoiStateType)
    (basis_l(op) == basis_r(op)) || throw(ArgumentError("Choi state must map between square operators in order to be converted to chi represenation"))
    bl, br = basis_l(basis_l(op)), basis_r(basis_l(op))
    foreach(_check_is_spinbasis, (bl, br))

    Nl, Nr = length(bl), length(br)
    Ul = ket_bra_to_pauli(Nl)
    Ur = Nl == Nr ? Ul : ket_bra_to_pauli(Nr)
    data = dagger(Ul)*op.data*Ur # TODO figure out normalization
    @assert isapprox(imag.(data), zero(data), atol=tol)
    Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
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
