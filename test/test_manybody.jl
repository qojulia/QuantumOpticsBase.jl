@testitem "test_manybody" begin
using Test
using QuantumOpticsBase
using Random, SparseArrays, LinearAlgebra

@testset "manybody" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

# Test `SortedVector`
ve = [3, 2, 1, 4, 5]
sv1 = QuantumOpticsBase.SortedVector(ve)
sv2 = QuantumOpticsBase.SortedVector(ve, Base.Reverse)
@test length(sv1) == 5
@test length(sv2) == 5
@test collect(sv1) == [1, 2, 3, 4, 5]
@test collect(sv2) == [5, 4, 3, 2, 1]
@test map(x -> x^2, sv1) == [1, 4, 9, 16, 25]
@test map(x -> x^2, sv2) == [25, 16, 9, 4, 1]
@test findfirst(iseven, sv1) == 2
@test findfirst(iseven, sv2) == 2

# Test state creation
Nmodes = 5
b = GenericBasis(Nmodes)
@test bosonstates(b, 1) == fermionstates(b, 1)
@test bosonstates(b, 2) != fermionstates(b, 2)
@test length(bosonstates(b, 1)) == Nmodes
@test bosonstates(b, 1) ∪ bosonstates(b, 2) == bosonstates(b, [1, 2])
@test fermionstates(b, 1) ∪ fermionstates(b, 2) == fermionstates(b, [1, 2])

@test ManyBodyBasis(b, bosonstates(b, 1)) == ManyBodyBasis(b, fermionstates(b, 1))
@test ManyBodyBasis(b, bosonstates(b, 2)) != ManyBodyBasis(b, fermionstates(b, 2))

@test collect(fermionstates(FermionBitstring, b, 3)) ==
    convert.(FermionBitstring, fermionstates(b, 3))
@test collect(bosonstates(Vector{Int}, b, 3)) ==
    convert.(Vector{Int}, bosonstates(b, 3))
b2 = GenericBasis(130)
@test collect(fermionstates(FermionBitstring{BigInt}, b2, 2)) ==
    convert.(FermionBitstring{BigInt}, fermionstates(b2, 2))

# Test basisstate
b_mb = ManyBodyBasis(b, bosonstates(b, 2))
psi_mb = basisstate(b_mb, [2, 0, 0, 0, 0])
op = dm(basisstate(b, 1))
@test onebodyexpect(op, psi_mb) ≈ 2
psi_mb = basisstate(b_mb, [1, 0, 1, 0, 0])
@test onebodyexpect(op, psi_mb) ≈ 1
@test_throws ArgumentError basisstate(b_mb, [1, 0, 0, 0, 0])

# Test creation operator
b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2]))
vac = basisstate(b_mb, [0, 0, 0, 0, 0])
at1 = create(b_mb, 1)
at2 = create(b_mb, 2)
at3 = create(b_mb, 3)
at4 = create(b_mb, 4)
at5 = create(b_mb, 5)
@test 1e-12 > D(at1*vac, basisstate(b_mb, [1, 0, 0, 0, 0]))
@test 1e-12 > D(at2*vac, basisstate(b_mb, [0, 1, 0, 0, 0]))
@test 1e-12 > D(at3*vac, basisstate(b_mb, [0, 0, 1, 0, 0]))
@test 1e-12 > D(at4*vac, basisstate(b_mb, [0, 0, 0, 1, 0]))
@test 1e-12 > D(at5*vac, basisstate(b_mb, [0, 0, 0, 0, 1]))

@test 1e-12 > D(at1*at1*vac, sqrt(2)*basisstate(b_mb, [2, 0, 0, 0, 0]))
@test 1e-12 > D(at1*at2*vac, basisstate(b_mb, [1, 1, 0, 0, 0]))
@test 1e-12 > norm(at1*basisstate(b_mb, [2, 0, 0, 0, 0]))

# Test annihilation operator
b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2]))
vac = basisstate(b_mb, [0, 0, 0, 0, 0])
a1 = destroy(b_mb, 1)
a2 = destroy(b_mb, 2)
a3 = destroy(b_mb, 3)
a4 = destroy(b_mb, 4)
a5 = destroy(b_mb, 5)
@test 1e-12 > D(a1*basisstate(b_mb, [1, 0, 0, 0, 0]), vac)
@test 1e-12 > D(a2*basisstate(b_mb, [0, 1, 0, 0, 0]), vac)
@test 1e-12 > D(a3*basisstate(b_mb, [0, 0, 1, 0, 0]), vac)
@test 1e-12 > D(a4*basisstate(b_mb, [0, 0, 0, 1, 0]), vac)
@test 1e-12 > D(a5*basisstate(b_mb, [0, 0, 0, 0, 1]), vac)

@test 1e-12 > D(a1*a1*basisstate(b_mb, [2, 0, 0, 0, 0]), sqrt(2)*vac)
@test 1e-12 > D(a1*a2*basisstate(b_mb, [1, 1, 0, 0, 0]), vac)
@test 1e-12 > norm(a1*vac)

# Test number operator
b = GenericBasis(3)
b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2, 3, 4]))
vac = basisstate(b_mb, [0, 0, 0])
n = number(b_mb)
n1 = number(b_mb, 1)
n2 = number(b_mb, 2)
n3 = number(b_mb, 3)

@test 1 ≈ expect(n2, basisstate(b_mb, [0, 1, 0]))
@test 2 ≈ expect(n3, basisstate(b_mb, [0, 0, 2]))
@test 0 ≈ expect(n1+n2+n3, vac)
@test 4 ≈ expect(n, basisstate(b_mb, [1, 3, 0]))

psi = randstate(b_mb)
rho = randoperator(b_mb)

n1_ = dm(basisstate(b, 1))
n2_ = dm(basisstate(b, 2))
n3_ = dm(basisstate(b, 3))

@test expect(n1, psi) ≈ onebodyexpect(n1_, psi)
@test expect(n2, psi) ≈ onebodyexpect(n2_, psi)
@test expect(n3, psi) ≈ onebodyexpect(n3_, psi)
@test expect(n1, rho) ≈ onebodyexpect(n1_, rho)
@test expect(n2, rho) ≈ onebodyexpect(n2_, rho)
@test expect(n3, rho) ≈ onebodyexpect(n3_, rho)

# Test transition operator
b = NLevelBasis(4)
b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2, 3]))
t23 = transition(b_mb, 2, 3)
t32 = transition(b_mb, 3, 2)

@test t23 == dagger(t32)
@test t32*basisstate(b_mb, [0, 1, 0, 0]) == basisstate(b_mb, [0, 0, 1, 0])
@test 1e-12 > D(t32*basisstate(b_mb, [0, 2, 1, 0]), 2*basisstate(b_mb, [0, 1, 2, 0]))
@test 1e-12 > D(number(b_mb, 2), transition(b_mb, 2, 2))

t22_33 = transition(b_mb, (2, 2), (3, 3))
d3 = destroy(b_mb, 3)
c2 = create(b_mb, 2)
@test t22_33 ≈ c2 * c2 * d3 * d3

# Test commutator and anticommutator
state = basisstate(b_mb, [1, 0, 0, 0])
a1 = destroy(b_mb, 1)
at2 = create(b_mb, 2)
at3 = create(b_mb, 3)
@test 1e-12 > D(a1 * at2 * state, at2 * a1 * state)
@test 1e-12 > D(at2 * at3 * state, at3 * at2 * state)

f_mb = ManyBodyBasis(b, fermionstates(b, [0, 1, 2]))
state = basisstate(f_mb, [1, 0, 0, 0])
a1 = destroy(f_mb, 1)
at2 = create(f_mb, 2)
at3 = create(f_mb, 3)
@test 1e-12 > D(a1 * at2 * state, -at2 * a1 * state)
@test 1e-12 > D(at2 * at3 * state, -at3 * at2 * state)

f_mb2 = ManyBodyBasis(b, fermionstates(FermionBitstring, b, [0, 1, 2]))
state = basisstate(f_mb2, [1, 0, 0, 0])
a1 = destroy(f_mb2, 1)
at2 = create(f_mb2, 2)
at3 = create(f_mb2, 3)
@test 1e-12 > D(a1 * at2 * state, -at2 * a1 * state)
@test 1e-12 > D(at2 * at3 * state, -at3 * at2 * state)

# Single particle operator in second quantization
b_single = GenericBasis(Nmodes)
b = ManyBodyBasis(b_single, bosonstates(b_single, [1, 2]))
x = randoperator(b_single)
y = randoperator(b_single)

X = manybodyoperator(b, x)
Y = manybodyoperator(b, y)

@test 1e-12 > D(X + Y, manybodyoperator(b, x + y))

x_ = sparse(x)
y_ = sparse(y)
@test 1e-12 > D(X, manybodyoperator(b, x_))
@test 1e-12 > D(Y, manybodyoperator(b, y_))
@test 1e-12 > D(X + Y, manybodyoperator(b, x_ + y_))

# Same for fermions
bf1 = ManyBodyBasis(b_single, fermionstates(b_single, [1, 2]))
bf2 = ManyBodyBasis(b_single, fermionstates(FermionBitstring, b_single, [1, 2]))

X1 = manybodyoperator(bf1, x)
X2 = manybodyoperator(bf2, x)
@test X1.data == X2.data
@test 1e-12 > D(X2, manybodyoperator(bf2, x_))

# Particle-particle interaction operator in second quantization
x = randoperator(b_single ⊗ b_single)
y = randoperator(b_single ⊗ b_single)

X = manybodyoperator(b, x)
Y = manybodyoperator(b, y)

@test 1e-12 > D(X + Y, manybodyoperator(b, x + y))

x_ = sparse(x)
y_ = sparse(y)
@test 1e-12 > D(X, manybodyoperator(b, x_))
@test 1e-12 > D(Y, manybodyoperator(b, y_))
@test 1e-12 > D(X + Y, manybodyoperator(b, x_ + y_))

# Same for fermions
X1 = manybodyoperator(bf1, x)
X2 = manybodyoperator(bf2, x)
Y2 = manybodyoperator(bf2, y)
@test X1.data == X2.data
@test 1e-12 > D(X2, manybodyoperator(bf2, x_))
@test 1e-12 > D(X2 + Y2, manybodyoperator(bf2, x + y))

# Test one-body expect
x = randoperator(b_single)
y = randoperator(b_single)
X = manybodyoperator(b, x)
Y = manybodyoperator(b, y)

psi = randstate(b)

@test onebodyexpect(x, psi) ≈ expect(X, dm(psi))
@test onebodyexpect(x, Y) ≈ expect(X, Y)
@test onebodyexpect(sparse(x), psi) ≈ expect(X, dm(psi))
@test onebodyexpect(sparse(x), Y) ≈ expect(X, Y)
@test onebodyexpect(x, [psi, Y]) == [onebodyexpect(x, psi), onebodyexpect(x, Y)]

@test_throws ArgumentError manybodyoperator(b_mb, x)
@test_throws ArgumentError onebodyexpect(X, psi)
@test_throws ArgumentError onebodyexpect(X, dm(psi))

end # testset
end