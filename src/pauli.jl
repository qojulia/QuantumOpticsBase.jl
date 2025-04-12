
const PauliTransferType = Operator{<:ChoiBasis,<:ChoiBasis}


# TODO this should maybe be exported?
# TODO also maybe more efficient to super-tensor product vec'd single qubit transformation
function _ketbra_to_pauli()
    b = SpinBasis(1//2)
    pvec(fn) = vec(fn(b)).data
    kb2p = sparse(hcat(map(pvec, [identityoperator, sigmax, sigmay, sigmaz])...)')
end

_Ukb2p = _ketbra_to_pauli()

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
function pauli(op::SuperOperatorType; tol=1e-9)
    bl, br = basis_l(op), basis_r(op)
    ((basis_l(bl) == basis_r(bl)) && (basis_l(br) == basis_r(br))) || throw(ArgumentError("Superoperator must map between square operators in order to be converted to pauli represenation"))

    for b in (basis_l(bl), basis_l(br))
        for i=1:length(b)
            (b[i] isa NLevelBasis) || throw(ArgumentError("Superoperator must be defined only systems composed of NLevelBasis to be converted to pauli representation"))
        end
    end
end
"""
