@testitem "test_sciml_broadcast_interfaces" begin
using Test
using QuantumOptics
using OrdinaryDiffEq

@testset "sciml interface" begin

# ket ODE problem
ℋ = SpinBasis(1//2)
↓ = spindown(ℋ)
t₀, t₁ = (0.0, pi)
σx = sigmax(ℋ)
iσx = im*σx
schrod!(dψ, ψ, p, t) = QuantumOptics.mul!(dψ, iσx, ψ)

ix = iσx.data
schrod_data!(dψ,ψ,p,t) = QuantumOptics.mul!(dψ, ix, ψ)
u0 = (↓).data

prob! = ODEProblem(schrod!, ↓, (t₀, t₁))
prob_data! = ODEProblem(schrod_data!, u0, (t₀, t₁))
sol = solve(prob!, DP5(); reltol = 1.0e-8, abstol = 1.0e-10, save_everystep=false)
sol_data = solve(prob_data!, DP5(); reltol = 1.0e-8, abstol = 1.0e-10, save_everystep=false)

@test sol[end].data ≈ sol_data[end] 

# dense operator ODE problem
σ₋ = sigmam(ℋ)
σ₊ = σ₋'
mhalfσ₊σ₋ = -σ₊*σ₋/2
ρ0 = dm(↓)
tmp = zero(ρ0)
function lind!(dρ,ρ,p,t)
	QuantumOptics.mul!(tmp, ρ, σ₊)
	QuantumOptics.mul!(dρ, σ₋, ρ)
	QuantumOptics.mul!(dρ,    ρ, mhalfσ₊σ₋, true, true)
	QuantumOptics.mul!(dρ, mhalfσ₊σ₋,    ρ, true, true)
	QuantumOptics.mul!(dρ,  iσx,    ρ, -ComplexF64(1),   ComplexF64(1))
	QuantumOptics.mul!(dρ,    ρ,  iσx,  true,   true)
	return dρ
end
m0 = ρ0.data
σ₋d = σ₋.data
σ₊d = σ₊.data
mhalfσ₊σ₋d = mhalfσ₊σ₋.data
tmpd = zero(m0)
function lind_data!(dρ,ρ,p,t)
    QuantumOptics.mul!(tmpd, ρ, σ₊d)
    QuantumOptics.mul!(dρ, σ₋d, ρ)
    QuantumOptics.mul!(dρ,  ρ, mhalfσ₊σ₋d, true, true)
    QuantumOptics.mul!(dρ, mhalfσ₊σ₋d, ρ, true, true)
    QuantumOptics.mul!(dρ,  ix,  ρ, -ComplexF64(1),   ComplexF64(1))
    QuantumOptics.mul!(dρ,  ρ,  ix,  true,   true)
    return dρ
end

prob! = ODEProblem(lind!, ρ0, (t₀, t₁))
prob_data! = ODEProblem(lind_data!, m0, (t₀, t₁))
sol = solve(prob!, DP5(); reltol = 1.0e-8, abstol = 1.0e-10, save_everystep=false)
sol_data = solve(prob_data!, DP5(); reltol = 1.0e-8, abstol = 1.0e-10, save_everystep=false)

@test sol[end].data ≈ sol_data[end]

end
end