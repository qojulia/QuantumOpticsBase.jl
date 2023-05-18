using Test
using QuantumOpticsBase
using SparseArrays, LinearAlgebra

@testset "metrics" begin

b1 = SpinBasis(1//2)
b2 = FockBasis(6)

psi1 = spinup(b1) ⊗ coherentstate(b2, 0.1)
psi2 = spindown(b1) ⊗ fockstate(b2, 2)

rho = tensor(psi1, dagger(psi1))
sigma = tensor(psi2, dagger(psi2))

# tracenorm
@test tracenorm(0*rho) ≈ 0.
@test tracenorm_h(0*rho) ≈ 0.
@test tracenorm_nh(0*rho) ≈ 0.

@test tracenorm(rho) ≈ 1.
@test tracenorm_h(rho) ≈ 1.
@test tracenorm_nh(rho) ≈ 1.

@test_throws ArgumentError tracenorm(sparse(rho))
@test_throws ArgumentError tracenorm_h(sparse(rho))
@test_throws ArgumentError tracenorm_nh(sparse(rho))

# tracedistance
@test tracedistance(rho, sigma) ≈ 1.
@test tracedistance_h(rho, sigma) ≈ 1.
@test tracedistance_nh(rho, sigma) ≈ 1.

@test tracedistance(rho, rho) ≈ 0.
@test tracedistance_h(rho, rho) ≈ 0.
@test tracedistance_nh(rho, rho) ≈ 0.

@test tracedistance(sigma, sigma) ≈ 0.
@test tracedistance_h(sigma, sigma) ≈ 0.
@test tracedistance_nh(sigma, sigma) ≈ 0.

@test_throws ArgumentError tracedistance(sparse(rho), sparse(rho))
@test_throws ArgumentError tracedistance_h(sparse(rho), sparse(rho))
@test_throws ArgumentError tracedistance_nh(sparse(rho), sparse(rho))

# tracedistance
@test tracedistance(rho, sigma) ≈ 1.
@test tracedistance(rho, rho) ≈ 0.
@test tracedistance(sigma, sigma) ≈ 0.

rho = spinup(b1) ⊗ dagger(coherentstate(b2, 0.1))
@test_throws ArgumentError tracedistance(rho, rho)
@test_throws ArgumentError tracedistance_h(rho, rho)

@test tracedistance_nh(rho, rho) ≈ 0.

# entropy_vn
rho_mix = dense(identityoperator(b1))/2.
@test entropy_vn(rho_mix)/log(2) ≈ 1
psi = coherentstate(FockBasis(20), 2.0)
@test isapprox(entropy_vn(psi), 0.0, atol=1e-8)

# entropy_renyi
rho_mix = dense(identityoperator(b1))/2.
@test entropy_renyi(rho_mix, 2)/log(2) ≈ 1
psi = coherentstate(FockBasis(20), 2.0)
@test isapprox(entropy_renyi(psi), 0.0, atol=1e-8)
@test_throws ArgumentError entropy_renyi(psi, 1)

# fidelity
rho = tensor(psi1, dagger(psi1))
@test fidelity(rho, rho) ≈ 1
@test 1e-20 > abs2(fidelity(rho, sigma))

# ptranspose
  
b1 = SpinBasis(1//2)
b2 = SpinBasis(1)
b3 = FockBasis(3)

# some tests for ptranspose only on randomly generated tripartite operators 
# consisting of `nterm` linear combinations of seperable operators
nterm = 3
coefs = rand(3)
As = [DenseOperator(b1, rand(2,2)) for i = 1 : nterm]
Bs = [DenseOperator(b2, rand(3,3)) for i = 1 : nterm]
Cs = [DenseOperator(b3, rand(4,4)) for i = 1 : nterm]

rho = sum([coefs[i]*As[i]⊗Bs[i]⊗Cs[i] for i = 1 : nterm]);

@test ptranspose(rho,(1,3)) == sum([coefs[i]*transpose(As[i])⊗Bs[i]⊗transpose(Cs[i]) for i = 1 : nterm])
@test ptranspose(rho,[2]) == transpose(ptranspose(rho,[1,3]))
@test ptranspose(rho,1) == sum([coefs[i]*transpose(As[i])⊗Bs[i]⊗Cs[i] for i = 1 : nterm])
  
e = spinup(b1)
g = spindown(b1)
psi3 = (e ⊗ g - g ⊗ e)/sqrt(2)

rho = dm(psi3)  

@test_throws MethodError ptranspose(e ⊗ dagger(psi1))
@test_throws MethodError ptranspose(dm(e))

rho = rho ⊗ dm(g)
@test PPT(rho, 1) == PPT(rho, 2) == false
@test PPT(rho, 3)

@test negativity(rho, 1) == negativity(rho, 2) ≈ 0.5
@test isapprox(negativity(rho, 3), 0.0, atol=1e-15)

@test logarithmic_negativity(rho, 1) == logarithmic_negativity(rho, 2) ≈ 1.0
@test isapprox(logarithmic_negativity(rho, 3), 0.0, atol=1e-15)

# Entanglement Entropy
b1 = SpinBasis(1//2)
psi = 1/sqrt(2)*(spinup(b1)⊗spindown(b1) + spindown(b1)⊗spinup(b1))
rho_ent = dm(psi)
rho_mix = DenseOperator(rho_ent.basis_l, diagm(ComplexF64[1.0,1.0,1.0,1.0]))
@test entanglement_entropy(rho_mix, 1) ≈ 0.0
@test entanglement_entropy(rho_ent, 1, entropy_vn) ≈ 2 * log(2)
@test entanglement_entropy(rho_ent, 1) ≈ 2 * entanglement_entropy(psi, 1)
@test_throws ArgumentError entanglement_entropy(rho_mix, (1,2))
@test_throws ArgumentError entanglement_entropy(rho_mix, 3)

q2 = PauliBasis(2)
CNOT = DenseOperator(q2, q2, diagm(0 => [1,1,0,0], 1 => [0,0,1], -1 => [0,0,1]))
CNOT_sop = SuperOperator(CNOT)
CNOT_chi = ChiMatrix(CNOT)
CNOT_ptm = PauliTransferMatrix(CNOT)

@test avg_gate_fidelity(CNOT_sop, CNOT_sop) == 1
@test avg_gate_fidelity(CNOT_chi, CNOT_chi) == 1
@test avg_gate_fidelity(CNOT_ptm, CNOT_ptm) == 1

@test_throws MethodError avg_gate_fidelity(CNOT_sop, CNOT_chi)
@test_throws MethodError avg_gate_fidelity(CNOT_sop, CNOT_ptm)
@test_throws MethodError avg_gate_fidelity(CNOT_chi, CNOT_ptm)

end # testset
