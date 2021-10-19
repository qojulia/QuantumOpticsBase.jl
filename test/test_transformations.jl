using Test
using QuantumOpticsBase
using Random, LinearAlgebra

@testset "transformation" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x::Ket, y::Ket) = norm(x-y)

# Test transformations
b_fock = FockBasis(20)
b_position = PositionBasis(-10, 10, 200)
Txn = transform(b_position, b_fock)
Tnx = transform(b_fock, b_position)
@test 1e-10 > D(dagger(Txn), Tnx)
@test 1e-10 > D(one(b_fock), Tnx*Txn)

x0 = 0.1
p0 = 0.3
α0 = (x0 + 1im*p0)/sqrt(2)
psi_n = coherentstate(b_fock, α0)
psi_x = gaussianstate(b_position, x0, p0, 1)
@test 1e-10 > D(psi_x, Txn*psi_n)

# Test different characteristic length
x0 = 0.0
p0 = 0.2
α0 = (x0 + 1im*p0)/sqrt(2)
σ0 = 0.7
Txn = transform(b_position, b_fock; x0=σ0)
Tnx = transform(b_fock, b_position; x0=σ0)
@test 1e-10 > D(dagger(Txn), Tnx)
@test 1e-10 > D(one(b_fock), Tnx*Txn)

psi_n = coherentstate(b_fock, α0)
psi_x = gaussianstate(b_position, x0/σ0, p0/σ0, σ0)
@test 1e-10 > D(psi_x, Txn*psi_n)

# Test with offset in FockBasis
b_fock = FockBasis(50,1)
b_position = PositionBasis(-10, 10, 300)

x0 = 6.1
p0 = 0.3
α0 = (x0 + 1im*p0)/sqrt(2)

psi_n = coherentstate(b_fock, α0)
psi_x = gaussianstate(b_position, x0, p0, 1)
@test isapprox(norm(psi_n), 1, atol=1e-8)

Txn = transform(b_position, b_fock)
Tnx = transform(b_fock, b_position)
@test 1e-10 > D(dagger(Txn), Tnx)
@test 1e-4 > D(psi_x, Txn*psi_n)
@test 1e-4 > D(psi_n, Tnx*psi_x)

end # testset
