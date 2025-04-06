using Test
using QuantumOpticsBase
using Random, SparseArrays, LinearAlgebra

@testset "fock" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
randstate(b) = normalize(Ket(b, rand(ComplexF64, length(b))))
randop(bl, br) = DenseOperator(bl, br, rand(ComplexF64, length(bl), length(br)))
randop(b) = randop(b, b)

basis = FockBasis(2)

# Test creation
@test basis.N == 2
@test length(basis) == 3
@test_throws ArgumentError FockBasis(-1)

# Test equality
@test FockBasis(2) == FockBasis(2)
@test FockBasis(2) != FockBasis(3)

# Test operators
@test number(basis) == SparseOperator(basis, sparse(Diagonal(ComplexF64[0, 1, 2])))
@test destroy(basis) == SparseOperator(basis, sparse(ComplexF64[0 1 0; 0 0 sqrt(2); 0 0 0]))
@test create(basis) == SparseOperator(basis, sparse(ComplexF64[0 0 0; 1 0 0; 0 sqrt(2) 0]))
@test number(basis) == dagger(number(basis))
@test create(basis) == dagger(destroy(basis))
@test destroy(basis) == dagger(create(basis))
@test 1e-15 > D(create(basis)*destroy(basis), number(basis))

# Test application onto statevectors
@test create(basis)*fockstate(basis, 0) == fockstate(basis, 1)
@test create(basis)*fockstate(basis, 1) == sqrt(2)*fockstate(basis, 2)
@test dagger(fockstate(basis, 0))*destroy(basis) == dagger(fockstate(basis, 1))
@test dagger(fockstate(basis, 1))*destroy(basis) == sqrt(2)*dagger(fockstate(basis, 2))

@test destroy(basis)*fockstate(basis, 1) == fockstate(basis, 0)
@test destroy(basis)*fockstate(basis, 2) == sqrt(2)*fockstate(basis, 1)
@test dagger(fockstate(basis, 1))*create(basis) == dagger(fockstate(basis, 0))
@test dagger(fockstate(basis, 2))*create(basis) == sqrt(2)*dagger(fockstate(basis, 1))

# Test displacement operator
b = FockBasis(30)
alpha = complex(0.5, 0.3)
d = displace(b, alpha)
@test 1e-12 > D(d*dagger(d), identityoperator(b))
@test 1e-12 > D(dagger(d)*d, identityoperator(b))
@test 1e-12 > D(dagger(d), displace(b, -alpha))
@test 1e-15 > norm(coherentstate(b, alpha) - displace(b, alpha)*fockstate(b, 0))
alpha = 5
@test coherentstate(b, alpha) ≈ displace_analytical(b, alpha)*fockstate(b, 0)

# Test squeezing operator
b = FockBasis(30)
z = complex(0.5, 0.3)
s = squeeze(b, z)
@test 1e-12 > D(s*dagger(s), identityoperator(b))
@test 1e-12 > D(dagger(s)*s, identityoperator(b))
@test 1e-12 > D(dagger(s), squeeze(b, -z))

α = complex(rand(0.0:0.1:2.0), rand(0.0:0.1:2.0))
for ofs in 0:3
    b = FockBasis(100, ofs)
    D = displace_analytical(b, α).data
    D̂ = displace(b, α).data
    
    chunk = 20
    imin = ofs > 0 ? chunk : 1
    imax = imin + 10
    @test D[imin:imax,imin:imax] ≈ D̂[imin:imax,imin:imax]

    m = rand(ofs:100)
    n = rand(ofs:100)
    @test D[m-ofs + 1, n-ofs + 1] == displace_analytical(α, m, n)
end

# Test Fock states
b = FockBasis(5)
@test expect(number(b), fockstate(b, 3)) == complex(3.)

# Test coherent states
b = FockBasis(100)
alpha = complex(3.)
a = destroy(b)
n = number(b)
psi = coherentstate(b, alpha)
rho = dm(psi)

@test 1e-14 > norm(expect(a, psi) - alpha)
@test 1e-14 > norm(expect(a, rho) - alpha)
@test 1e-13 > abs(variance(n, psi) - abs(alpha)^2)
@test 1e-13 > abs(variance(n, rho) - abs(alpha)^2)

# Test FockBasis with offset
b_off = FockBasis(100,4)
@test_throws AssertionError fockstate(b_off, 0)
n = 55
psi = fockstate(b_off, n)
@test expect(number(b_off), psi)==n==expect(create(b_off)*destroy(b_off), psi)

alpha = 5
psi = coherentstate(b, alpha)
psi_off = coherentstate(b_off, alpha)
@test psi.data[b_off.offset+1:end] == psi_off.data
@test isapprox(norm(psi_off), 1, atol=1e-7)

end # testset
