using Test
using QuantumOpticsBase
using LinearAlgebra, Random


@testset "operators-lazysum" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

# Test creation
@test_throws ArgumentError LazySum()
@test_throws ArgumentError LazySum([1., 2.], [randoperator(b_l)])
@test_throws QuantumOpticsBase.IncompatibleBases LazySum(randoperator(b_l, b_r), sparse(randoperator(b_l, b_l)))
@test_throws QuantumOpticsBase.IncompatibleBases LazySum(randoperator(b_l, b_r), sparse(randoperator(b_r, b_r)))

# Test copy
op1 = 2*LazySum(randoperator(b_l, b_r), sparse(randoperator(b_l, b_r)))
op2 = copy(op1)
@test op1 == op2
@test isequal(op1, op2)
@test !(op1 === op2)
op2.operators[1].data[1,1] = complex(10.)
@test op1.operators[1].data[1,1] != op2.operators[1].data[1,1]
op2.factors[1] = 3.
@test op2.factors[1] != op1.factors[1]

# Test dense & sparse
op1 = randoperator(b_l, b_r)
op2 = sparse(randoperator(b_l, b_r))
@test 0.1*op1 == dense(LazySum([0.1], (op1,)))
@test 0.3*op2 == sparse(LazySum([0.3], (op2,)))
@test 0.1*sparse(op1) + 0.3*op2 == sparse(LazySum([0.1, 0.3], (op1, op2)))
@test 0.1*op1 + 0.3*dense(op2) == dense(LazySum([0.1, 0.3], (op1, op2)))
@test 0.1*sparse(op1) + 0.3*op2 == sparse(LazySum([0.1, 0.3], (op1, op2)))

# Test embed
for T in (Float32, Float64, ComplexF32, ComplexF64)
    x1 = randoperator(T, b1a,b1b)
    y1 = randoperator(T, b1a,b1b)
    xy1 = LazySum(T[1., 2.], (x1, y1))
    x = LazySum(T[1.], (embed(b_l, b_r, 1, x1),))
    y = LazySum(T[2.], (embed(b_l, b_r, 1, y1),))
    @test eltype(x) == T
    @test eltype(y) == T
    xy = x + y
    @test embed(b_l, b_r, [1], xy1) == xy
    @test eltype(xy) == T
end


# Arithmetic operations
# =====================
op1a = randoperator(b_l, b_r)
op1b = randoperator(b_l, b_r)
op2a = randoperator(b_l, b_r)
op2b = randoperator(b_l, b_r)
op3a = randoperator(b_l, b_r)
op1 = LazySum([0.1, 0.3], (op1a, sparse(op1b)))
op1_ = 0.1*op1a + 0.3*op1b
op2 = LazySum([0.7, 0.9], [sparse(op2a), op2b])
op2_ = 0.7*op2a + 0.9*op2b
op3 = LazySum(op3a)
op3_ = op3a

x1 = Ket(b_r, rand(ComplexF64, length(b_r)))
x2 = Ket(b_r, rand(ComplexF64, length(b_r)))
xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Addition
@test_throws QuantumOpticsBase.IncompatibleBases op1 + dagger(op2)
@test 1e-14 > D(op1+op2, op1_+op2_)
@test 1e-14 > D(op1+op2_, op1_+op2_)
@test 1e-14 > D(op1_+op2, op1_+op2_)

# Subtraction
@test_throws QuantumOpticsBase.IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1 - op2, op1_ - op2_)
@test 1e-14 > D(op1 - op2_, op1_ - op2_)
@test 1e-14 > D(op1_ - op2, op1_ - op2_)
@test 1e-14 > D(op1 + (-op2), op1_ - op2_)
@test 1e-14 > D(op1 + (-1*op2), op1_ - op2_)

# Test multiplication
@test_throws ArgumentError op1*op2
@test LazySum([0.1, 0.1], (op1a, op2a)) == LazySum(op1a, op2a)*0.1
@test LazySum([0.1, 0.1], (op1a, op2a)) == 0.1*LazySum(op1a, op2a)
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), (op1_+op2_)*(x1+0.3*x2))
@test 1e-12 > D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))

## Test multiplication with LazySum that has no elements
@test iszero( LazySum(b_r, b_l) * op1a )
@test iszero( op1a * LazySum(b_r, b_l) )
@test iszero( LazySum(b_l, b_r) * x1 )
@test iszero( xbra1 * LazySum(b_l, b_r) )
@test_throws DimensionMismatch LazySum(FockBasis(2), NLevelBasis(2)) * randoperator(NLevelBasis(4), GenericBasis(2)) # save Basis with different size
@test_throws DimensionMismatch randoperator(GenericBasis(1), FockBasis(3)) * LazySum(FockBasis(1), NLevelBasis(2))
@test_throws DimensionMismatch LazySum(FockBasis(2), NLevelBasis(2)) * randstate(NLevelBasis(7))
@test_throws DimensionMismatch randstate(FockBasis(3))' * LazySum(FockBasis(1), NLevelBasis(2))

## multiplication with Operator of AbstractMatrix
LSop = LazySum(randoperator(b1a^2)) # AbstractOperator
LSop_s = LazySum(sparse(randoperator(b1a^2)))
hermitian_op = Operator(basis(LSop), Hermitian(randn(ComplexF64,length(basis(LSop)),length(basis(LSop))))) # Hermitian
symmetric_op = Operator(basis(LSop), Symmetric(randn(ComplexF64,length(basis(LSop)),length(basis(LSop))))) # Symmetric
adjoint_op = randoperator(basis(LSop))' # Adjoint
real_op = Operator(basis(LSop), real(adjoint_op.data)) # real
ops_tuple = (symmetric_op,hermitian_op,adjoint_op)
### Test
@test ops_tuple.*(LSop,) == dense.(ops_tuple).*(LSop,)
@test (LSop,).*ops_tuple == (LSop,).*dense.(ops_tuple)
### test with sparse
@test all(isapprox.(sparse.(ops_tuple).*(LSop,) , dense.(ops_tuple).*(LSop,), atol=1e-13))
@test all(isapprox.((LSop,).*sparse.(ops_tuple) , (LSop,).*dense.(ops_tuple), atol=1e-13))
@test all(isapprox.(dense.(sparse.(ops_tuple).*(LSop_s,)) , dense.(ops_tuple).*(LSop_s,), atol=1e-13))
@test all(isapprox.(dense.((LSop_s,).*sparse.(ops_tuple)) , (LSop_s,).*dense.(ops_tuple), atol=1e-13))
### test real valued op with AbstractOperator
@test isapprox(LSop*real_op*LSop , LSop*Operator(basis(real_op), complex.(real_op.data))*LSop, atol=1e-13)

# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Test tuples vs. vectors
@test (op1+op1).operators isa Tuple
@test (op1+op2).operators isa Tuple
@test (op2+op1).operators isa Tuple
@test (op2+op2).operators isa Vector

# Test identityoperator
Idense = identityoperator(DenseOpType, b_r)
id = identityoperator(LazySum, b_r)
@test isa(id, LazySum)
@test dense(id) == Idense
@test 1e-11 > D(id*x1, x1)

Idense = identityoperator(DenseOpType, b_l)
id = identityoperator(LazySum, b_l)
@test isa(id, LazySum)
@test dense(id) == Idense
@test 1e-11 > D(xbra1*id, xbra1)

# Test tr and normalize
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op3 = randoperator(b_l)
op = LazySum([0.1, 0.3, 1.2], (op1, op2, op3))
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

@test tr(op_) ≈ tr(op)
op_normalized = normalize(op)
@test tr(op_) ≈ tr(op)
@test 1 ≈ tr(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test tr(op) != tr(op_copy)
@test 1 ≈ tr(op_copy)
@test op_copy === normalize!(op_copy)


# Test partial tr
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op3 = randoperator(b_l)
op123 = LazySum([0.1, 0.3, 1.2], (op1, op2, op3))
op123_v = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op123_ = 0.1*op1 + 0.3*op2 + 1.2*op3

@test 1e-14 > D(ptrace(op123_, 3), ptrace(op123, 3))
@test 1e-14 > D(ptrace(op123_, 2), ptrace(op123, 2))
@test 1e-14 > D(ptrace(op123_, 1), ptrace(op123, 1))

@test 1e-14 > D(ptrace(op123_, [2,3]), ptrace(op123, [2,3]))
@test 1e-14 > D(ptrace(op123_, [1,3]), ptrace(op123, [1,3]))
@test 1e-14 > D(ptrace(op123_, [1,2]), ptrace(op123, [1,2]))

@test 1e-14 > D(ptrace(op123_v, [1,2]), ptrace(op123, [1,2]))

@test_throws ArgumentError ptrace(op123, [1,2,3])

# Test expect
state = Ket(b_l, rand(ComplexF64, length(b_l)))
@test expect(op123, state) ≈ expect(op123_, state)

state = DenseOperator(b_l, b_l, rand(ComplexF64, length(b_l), length(b_l)))
@test expect(op123, state) ≈ expect(op123_, state)

# Permute systems
op1a = randoperator(b1a)
op2a = randoperator(b2a)
op3a = randoperator(b3a)
op1b = randoperator(b1a)
op2b = randoperator(b2a)
op3b = randoperator(b3a)
op1c = randoperator(b1a)
op2c = randoperator(b2a)
op3c = randoperator(b3a)
op123a = op1a⊗op2a⊗op3a
op123b = op1b⊗op2b⊗op3b
op123c = op1c⊗op2c⊗op3c
op = LazySum([0.3, 0.7, 1.2], [op123a, sparse(op123b), op123c])
op_ = 0.3*op123a + 0.7*op123b + 1.2*op123c

@test 1e-14 > D(permutesystems(op, [1, 3, 2]), permutesystems(op_, [1, 3, 2]))
@test 1e-14 > D(permutesystems(op, [2, 1, 3]), permutesystems(op_, [2, 1, 3]))
@test 1e-14 > D(permutesystems(op, [2, 3, 1]), permutesystems(op_, [2, 3, 1]))
@test 1e-14 > D(permutesystems(op, [3, 1, 2]), permutesystems(op_, [3, 1, 2]))
@test 1e-14 > D(permutesystems(op, [3, 2, 1]), permutesystems(op_, [3, 2, 1]))


# Test gemv
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_l, b_r)
op3 = randoperator(b_l, b_r)
op = LazySum([0.1, 0.3, 1.2], (op1, op2, op3))
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

zero_op = LazySum(b_l, b_r)
zero_op_ = sparse(zero_op)
@test dense(zero_op) == zero_op_

state = Ket(b_r, rand(ComplexF64, length(b_r)))
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,zero_op,state,complex(1.),complex(0.))
@test 1e-13 > D(result, zero_op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,zero_op,state,alpha,beta)
@test 1e-13 > D(result, beta*result_)

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,zero_op,complex(1.),complex(0.))
@test 1e-13 > D(result, state*zero_op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_l, b_r)
op3 = randoperator(b_l, b_r)
op = LazySum([0.1, 0.3, 1.2], (op1, op2, op3))
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

state = randoperator(b_r, b_r)
result_ = randoperator(b_l, b_r)
result = NaN * deepcopy(result_)  # with beta=0, NaNs should be killed
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,zero_op,state,complex(1.),complex(0.))
@test 1e-12 > D(result, zero_op_*state)

alpha = complex(1.5)
beta = complex(2.1)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,zero_op,state,alpha,beta)
@test 1e-12 > D(result, beta*result_)

state = randoperator(b_l, b_l)
result_ = randoperator(b_l, b_r)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-12 > D(result, state*op_)

result = NaN * deepcopy(result_)  # with beta=0, NaNs should be killed
QuantumOpticsBase.mul!(result,state,zero_op,complex(1.),complex(0.))
@test 1e-12 > D(result, state*zero_op_)

alpha = complex(1.5)
beta = complex(2.1)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,zero_op,alpha,beta)
@test 1e-12 > D(result, beta*result_)

end # testset
