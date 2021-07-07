"""
    randstate([T=ComplexF64,] basis)

Calculate a random normalized ket state.
"""
function randstate(::Type{T}, b::Basis) where T
    psi = Ket(b, rand(T, length(b)))
    normalize!(psi)
    psi
end
randstate(b) = randstate(ComplexF64, b)

"""
    randoperator([T=ComplexF64,] b1[, b2])

Calculate a random unnormalized dense operator.
"""
randoperator(::Type{T}, b1::Basis, b2::Basis) where T = DenseOperator(b1, b2, rand(T, length(b1), length(b2)))
randoperator(b1::Basis, b2::Basis) = randoperator(ComplexF64, b1, b2)
randoperator(::Type{T}, b::Basis) where T = randoperator(T, b, b)
randoperator(b) = randoperator(ComplexF64, b)

"""
    thermalstate(H,T)

Thermal state ``exp(-H/T)/Tr[exp(-H/T)]``.
"""
function thermalstate(H,T)
    return normalize(exp(-dense(H)/T))
end

"""
    coherentthermalstate([C=ComplexF64,] basis::FockBasis,H,T,alpha)

Coherent thermal state ``D(α)exp(-H/T)/Tr[exp(-H/T)]D^†(α)``.
"""
function coherentthermalstate(::Type{C},basis::B,H::AbstractOperator{B,B},T,alpha) where {C,B<:FockBasis}
    D = displace(C,basis,alpha)
    return D*thermalstate(H,T)*dagger(D)
end
coherentthermalstate(basis::B,H::AbstractOperator{B,B},T,alpha) where B<:FockBasis = coherentthermalstate(ComplexF64,basis,H,T,alpha)

"""
    phase_average(rho)

Returns the phase-average of ``ρ`` containing only the diagonal elements.
"""
function phase_average(rho)
    return Operator(basis(rho),diagm(0 => diag(rho.data)))
end

"""
    passive_state(rho,IncreasingEigenenergies=true)

Passive state ``π`` of ``ρ``. IncreasingEigenenergies=true means that higher indices correspond to higher energies.
"""
function passive_state(rho,IncreasingEigenenergies=true)
    return DenseOperator(basis(rho),diagm(0 => sort!(abs.(eigvals(rho.data)),rev=IncreasingEigenenergies)))
end
