"""
    randstate(basis)

Calculate a random normalized ket state.
"""
function randstate(b)
    psi = Ket(b, rand(ComplexF64, length(b)))
    normalize!(psi)
    psi
end

"""
    randoperator(b1[, b2])

Calculate a random unnormalized dense operator.
"""
randoperator(b1, b2) = DenseOperator(b1, b2, rand(ComplexF64, length(b1), length(b2)))
randoperator(b) = randoperator(b, b)

"""
    thermalstate(H,T)

Thermal state ``exp(-H/T)/Tr[exp(-H/T)]``.
"""
function thermalstate(H,T)
    return normalize(exp(-dense(H)/T))
end

"""
    coherentthermalstate(basis::FockBasis,H,T,alpha)

Coherent thermal state ``D(α)exp(-H/T)/Tr[exp(-H/T)]D^†(α)``.
"""
function coherentthermalstate(basis::B,H::AbstractOperator{B,B},T,alpha) where B<:FockBasis
    D = displace(basis,alpha)
    return D*thermalstate(H,T)*dagger(D)
end

"""
    phase_average(rho)

Returns the phase-average of ``ρ`` containing only the diagonal elements.
"""
function phase_average(rho)
    return Operator(basis(rho),diagm(0 => diag(rho.data)))
end

"""
    passive_state(rho,IncreasingEigenenergies::Bool=true)

Passive state ``π`` of ``ρ``. IncreasingEigenenergies=true means that higher indices correspond to higher energies.
"""
function passive_state(rho,IncreasingEigenenergies=true)
    return DenseOperator(basis(rho),diagm(0 => sort!(abs.(eigvals(rho.data)),rev=IncreasingEigenenergies)))
end
