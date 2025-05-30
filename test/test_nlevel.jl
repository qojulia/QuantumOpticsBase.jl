@testitem "nlevel" begin
using QuantumOpticsBase
using LinearAlgebra

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))

N = 3
b = NLevelBasis(N)

# Test basis equality
@test b == NLevelBasis(N)
@test b != NLevelBasis(N+1)
@test_throws ArgumentError NLevelBasis(0)

# Test transition operator
@test_throws BoundsError transition(b, 0, 1)
@test_throws BoundsError transition(b, N+1, 1)
@test_throws BoundsError transition(b, 1, 0)
@test_throws BoundsError transition(b, 1, N+1)
@test dense(transition(b, 2, 1)) == basisstate(b, 2) ⊗ dagger(basisstate(b, 1))

# Test nlevel states
@test_throws BoundsError nlevelstate(b, 0)
@test_throws BoundsError nlevelstate(b, N+1)
@test norm(nlevelstate(b, 1)) == 1.
@test norm(dagger(nlevelstate(b, 1))*nlevelstate(b, 2)) == 0.
@test norm(dagger(nlevelstate(b, 1))*transition(b, 1, 2)*nlevelstate(b, 2)) == 1.

for N=[2, 3, 4, 5]
    b = NLevelBasis(N)
    I = identityoperator(b)
    Zero = SparseOperator(b)
    px = paulix(b)
    pz = pauliz(b)

    @test 1e-14 > abs(tr(px))
    @test 1e-14 > abs(tr(pz))
    @test 1e-14 > D(px^N, I)
    @test 1e-14 > D(pz^N, I)
    for m=2:N
        @test 1e-14 > D(px^m, paulix(b,m))
        @test 1e-14 > D(pz^m, pauliz(b,m))
    end

    for m=1:N,n=1:N
        @test 1e-14 > D(px^m * pz^n, exp(2π*1im*m*n/N) * pz^n * px^m)
    end
end


# Test special relations for qubit pauli
b = NLevelBasis(2)
I = identityoperator(b)
Zero = SparseOperator(b)
px = paulix(b)
py = pauliy(b)
pz = pauliz(b)

antikommutator(x, y) = x*y + y*x

@test 1e-14 > D(antikommutator(px, px), 2*I)
@test 1e-14 > D(antikommutator(px, py), Zero)
@test 1e-14 > D(antikommutator(px, pz), Zero)
@test 1e-14 > D(antikommutator(py, px), Zero)
@test 1e-14 > D(antikommutator(py, py), 2*I)
@test 1e-14 > D(antikommutator(py, pz), Zero)
@test 1e-14 > D(antikommutator(pz, px), Zero)
@test 1e-14 > D(antikommutator(pz, py), Zero)
@test 1e-14 > D(antikommutator(pz, pz), 2*I)

# Test if involutory for spin 1/2
@test 1e-14 > D(px*px, I)
@test 1e-14 > D(py*py, I)
@test 1e-14 > D(pz*pz, I)
@test 1e-14 > D(-1im*px*py*pz, I)

end # testset
