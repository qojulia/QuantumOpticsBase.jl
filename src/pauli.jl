using TransmuteDims

const PauliTransferType = Operator{<:PauliBasis,<:PauliBasis}

# TODO should either of these functions be cached?
function _pauli_to_ketbra1()
    b = SpinBasis(1//2)
    vec_it(fn) = vec(fn(b).data)
    p2kb = hcat(map(vec_it, [identityoperator, sigmax, sigmay, sigmaz])...)
end

function _ketbra_to_pauli(N)
    T = kron((_pauli_to_ketbra1 for _=1:N)...)
    T = reshape(T, Tuple(4 for _=1:2N))
    T = PermutedDimsArray(T, ((2i-1 for i=1:N)..., (2i for i=1:N)...))
    reshape(T, (4^N, 4^N))
end

function _op_to_pauli(Nl, Nr, data)
    Ul = ket_bra_to_pauli(Nl)
    Ur = Nl == Nr ? Ul : _ketbra_to_pauli(Nr)
    data = dagger(Ul) * data * Ur # TODO figure out normalization
    @assert isapprox(imag.(data), zero(data), atol=tol)
    Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
end

"""
    comp_to_pauli(N)

Creates a superoperator which changes from the computational `SpinBasis(1//2)`
to the `PauliBasis()` over `N` qubits.
"""
function comp_to_pauli(N)
    T = kron((_pauli_to_ketbra1 for _=1:N)...)
    T = reshape(T, Tuple(4 for _=1:2N))
    T = PermutedDimsArray(T, ((2i-1 for i=1:N)..., (2i for i=1:N)...))
    sb = SpinBasis(1//2)^N
    Operator(PauliBasis()^N, KetBraBasis(sb, sb), reshape(T, (4^N, 4^N))
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
    Ur = Nl == Nr ? Ul : _ketbra_to_pauli(Nr)
    data = dagger(Ul)*op.data*Ur # TODO figure out normalization
    @assert isapprox(imag.(data), zero(data), atol=tol)
    Operator(PauliBasis()^Nl, PauliBasis()^Nr, real.(data))
end

function chi(op::ChoiStateType; tol=1e-9)
    (basis_l(op) == basis_r(op)) || throw(ArgumentError("Choi state must map between square operators in order to be converted to chi represenation"))

    bl, br = basis_l(basis_l(op)), basis_r(basis_l(op))
    for b in (bl, br)
        for i=1:length(b)
            (b[i] isa SpinBasis && dimension(b[i]) == 2) || throw(ArgumentError("Choi state must be over systems composed of SpinBasis(1//2) to be converted to chi representation"))
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
