using Test
using QuantumOpticsBase
import QuantumInterface: IncompatibleBases
using LinearAlgebra, Random


@testset "operators-lazyproduct" begin

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
@test_throws ArgumentError LazyProduct()
@test_throws IncompatibleBases LazyProduct(randoperator(b_l, b_r), randoperator(b_l, b_r))
@test_throws IncompatibleBases LazyProduct(randoperator(b_l, b_r), sparse(randoperator(b_l, b_r)))

# Test copy
op1 = 2*LazyProduct(randoperator(b_l, b_r), sparse(randoperator(b_r, b_l)))
op2 = copy(op1)
@test op1 == op2
@test isequal(op1, op2)
@test !(op1 === op2)
op2.operators[1].data[1,1] = complex(10.)
@test op1.operators[1].data[1,1] != op2.operators[1].data[1,1]
op2.factor = 3.
@test op2.factor != op1.factor

@test QuantumOpticsBase.is_const(op1)

# Test dense & sparse
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_r, b_l)
@test 0.1*(op1*op2) == dense(LazyProduct([sparse(op1), sparse(op2)], 0.1))
@test 0.1*(sparse(op1)*sparse(op2)) == sparse(LazyProduct([op1, op2], 0.1))

# Arithmetic operations
# =====================
op1a = randoperator(b_l, b_r)
op1b = randoperator(b_r, b_l)
op2a = randoperator(b_l, b_r)
op2b = randoperator(b_r, b_l)
op3a = randoperator(b_l, b_l)
op3b = randoperator(b_r, b_r)

op1 = LazyProduct([op1a, sparse(op1b)])*0.1
op1_ = 0.1*(op1a*op1b)
op2 = LazyProduct([sparse(op2a), op2b], 0.3)
op2_ = 0.3*(op2a*op2b)
op3 = LazyProduct(op3a)
op3_ = op3a
op4 = LazyProduct(op3b)
op4_ = op3b

x1 = Ket(b_l, rand(ComplexF64, length(b_l)))
x2 = Ket(b_l, rand(ComplexF64, length(b_l)))
xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))
xbra2 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Test Addition
@test_throws IncompatibleBases op1 + dagger(op4)
@test 1e-14 > D(-op1_, -op1)
@test 1e-14 > D(op1+op2, op1_+op2_)
@test 1e-14 > D(op1+op2_, op1_+op2_)
@test 1e-14 > D(op1_+op2, op1_+op2_)
#Check for unallowed addition:
@test_throws IncompatibleBases LazyProduct([op1a, sparse(op1a)])+LazyProduct([sparse(op2b), op2b], 0.3)

# Test Subtraction
@test_throws IncompatibleBases op1 - dagger(op4)
@test 1e-14 > D(op1 - op2, op1_ - op2_)
@test 1e-14 > D(op1 - op2_, op1_ - op2_)
@test 1e-14 > D(op1_ - op2, op1_ - op2_)
@test 1e-14 > D(op1 + (-op2), op1_ - op2_)
@test 1e-14 > D(op1 + (-1*op2), op1_ - op2_)
#Check for unallowed subtraction:
@test_throws IncompatibleBases LazyProduct([op1a, sparse(op1a)])-LazyProduct([sparse(op2b), op2b], 0.3)


# Test multiplication
@test_throws IncompatibleBases op1a*op1a
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_)
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-12 > D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))
@test 0.3*LazyProduct(op1, sparse(op2)) == LazyProduct([op1, sparse(op2)], 0.3)
@test 0.3*LazyProduct(op1)*LazyProduct(sparse(op2)) == LazyProduct([op1, sparse(op2)], 0.3)

# Test division
@test 1e-14 > D(op1/7, op1_/7)

#Test Tensor product
op = op1a ⊗ op1
op_ = op1a ⊗ op1_
@test 1e-11 > D(op,op_)
op = op1 ⊗ op2a
op_ = op1_ ⊗ op2a 
@test 1e-11 > D(op,op_)

# Test identityoperator
Idense = identityoperator(DenseOpType, b_l)
id = identityoperator(LazyProduct, b_l)
@test isa(id, LazyProduct)
@test dense(id) == Idense
@test 1e-11 > D(id*x1, x1)
@test 1e-11 > D(xbra1*id, xbra1)

# Test tr and normalize
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op = LazyProduct(op1, op2)
@test_throws ArgumentError tr(op)
@test_throws ArgumentError ptrace(op, [1, 2])
@test_throws ArgumentError normalize(op)
@test_throws ArgumentError normalize!(op)

# Test expect
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op = 0.3*LazyProduct(op1, sparse(op2))
op_ = 0.3*op1*op2

state = Ket(b_l, rand(ComplexF64, length(b_l)))
@test expect(op, state) ≈ expect(op_, state)

state = DenseOperator(b_l, b_l, rand(ComplexF64, length(b_l), length(b_l)))
@test expect(op, state) ≈ expect(op_, state)

# Permute systems
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op3 = randoperator(b_l)
op = 0.3*LazyProduct(op1, op2, sparse(op3))
op_ = 0.3*op1*op2*op3

@test 1e-14 > D(permutesystems(op, [1, 3, 2]), permutesystems(op_, [1, 3, 2]))
@test 1e-14 > D(permutesystems(op, [2, 1, 3]), permutesystems(op_, [2, 1, 3]))
@test 1e-14 > D(permutesystems(op, [2, 3, 1]), permutesystems(op_, [2, 3, 1]))
@test 1e-14 > D(permutesystems(op, [3, 1, 2]), permutesystems(op_, [3, 1, 2]))
@test 1e-14 > D(permutesystems(op, [3, 2, 1]), permutesystems(op_, [3, 2, 1]))


# Test gemv
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_r, b_l)
op3 = randoperator(b_l, b_r)
op4 = randoperator(b_r, b_l)
for N=1:3
    op = LazyProduct([op1, sparse(op2), op3, op4][1:N], 0.2)
    op_ = 0.2*prod([op1,op2,op3,op4][1:N])

    state = randstate(iseven(N) ? b_l : b_r)
    result_ = randstate(b_l)
    result = copy(result_)
    QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
    @test 1e-11 > D(result, op_*state)

    result = copy(result_)
    alpha = complex(1.5)
    beta = complex(2.1)
    QuantumOpticsBase.mul!(result,op,state,alpha,beta)
    @test 1e-11 > D(result, alpha*op_*state + beta*result_)

    result = copy(result_)
    QuantumOpticsBase.mul!(result,op,state,0,beta)
    @test 1e-11 > D(result, beta*result_)

    state = Bra(b_l, rand(ComplexF64, length(b_l)))
    result_ = randstate(iseven(N) ? b_l : b_r)'
    result = copy(result_)
    QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
    @test 1e-11 > D(result, state*op_)

    result = copy(result_)
    alpha = complex(1.5)
    beta = complex(2.1)
    QuantumOpticsBase.mul!(result,state,op,alpha,beta)
    @test 1e-11 > D(result, alpha*state*op_ + beta*result_)

    result = copy(result_)
    QuantumOpticsBase.mul!(result,state,op,0,beta)
    @test 1e-11 > D(result, beta*result_)

    # Test gemm
    state = randoperator(iseven(N) ? b_l : b_r, b_r)
    result_ = randoperator(b_l, b_r)
    result = copy(result_)
    QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
    @test 1e-11 > D(result, op_*state)

    result = copy(result_)
    alpha = complex(1.5)
    beta = complex(2.1)
    QuantumOpticsBase.mul!(result,op,state,alpha,beta)
    @test 1e-11 > D(result, alpha*op_*state + beta*result_)

    result = copy(result_)
    QuantumOpticsBase.mul!(result,op,state,0,beta)
    @test 1e-11 > D(result, beta*result_)

    state = randoperator(b_l, b_l)
    result_ = randoperator(b_l, iseven(N) ? b_l : b_r)
    result = copy(result_)
    QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
    @test 1e-11 > D(result, state*op_)

    result = copy(result_)
    alpha = complex(1.5)
    beta = complex(2.1)
    QuantumOpticsBase.mul!(result,state,op,alpha,beta)
    @test 1e-11 > D(result, alpha*state*op_ + beta*result_)

    result = copy(result_)
    QuantumOpticsBase.mul!(result,state,op,0,beta)
    @test 1e-11 > D(result, beta*result_)
end

end # testset
