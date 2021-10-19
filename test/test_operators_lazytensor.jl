using Test
using QuantumOpticsBase
using LinearAlgebra, SparseArrays, Random

mutable struct test_lazytensor{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::Matrix{ComplexF64}
    test_lazytensor(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new{typeof(b1),typeof(b2)}(b1, b2, data) : throw(DimensionMismatch())
end
Base.eltype(::test_lazytensor) = ComplexF64

@testset "operators-lazytensor" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(6)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

op1 = randoperator(b1a, b1b)
op2 = randoperator(b2a, b2b)
op3 = randoperator(b3a, b3b)

# Test creation
@test_throws AssertionError LazyTensor(b_l, b_r, [1], (randoperator(b1a),))
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], (op1,))
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], (op1, sparse(randoperator(b_l, b_l))))
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], (randoperator(b_r, b_r), sparse(op2)))

# @test LazyTensor(b_l, b_r, [2, 1], [op2, op1]) == LazyTensor(b_l, b_r, [1, 2], [op1, op2])
x = randoperator(b2a)
@test LazyTensor(b_l, 2, x) == LazyTensor(b_l, b_l, [2], (x,))

# Test copy
x = 2*LazyTensor(b_l, b_r, [1,2], (randoperator(b1a, b1b), sparse(randoperator(b2a, b2b))))
x_ = copy(x)
@test x == x_
@test !(x === x_)
x_.operators[1].data[1,1] = complex(10.)
@test x.operators[1].data[1,1] != x_.operators[1].data[1,1]
x_.factor = 3.
@test x_.factor != x.factor
x_.indices[2] = 100
@test x_.indices != x.indices


# Test dense & sparse
I2 = identityoperator(b2a, b2b)
x = LazyTensor(b_l, b_r, [1, 3], (op1, sparse(op3)), 0.3)
@test 1e-12 > D(0.3*op1⊗dense(I2)⊗op3, dense(x))
@test 1e-12 > D(0.3*sparse(op1)⊗I2⊗sparse(op3), sparse(x))

# Test suboperators
@test QuantumOpticsBase.suboperator(x, 1) == op1
@test QuantumOpticsBase.suboperator(x, 3) == sparse(op3)
@test QuantumOpticsBase.suboperators(x, [1, 3]) == [op1, sparse(op3)]


# Arithmetic operations
# =====================
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I1 = dense(identityoperator(b1a, b1b))
I2 = dense(identityoperator(b2a, b2b))
I3 = dense(identityoperator(b3a, b3b))
op1 = LazyTensor(b_l, b_r, [1, 3], (subop1, sparse(subop3)), 0.1)
op1_ = 0.1*subop1 ⊗ I2 ⊗ subop3
op2 = LazyTensor(b_l, b_r, [2, 3], (sparse(subop2), subop3), 0.7)
op2_ = 0.7*I1 ⊗ subop2 ⊗ subop3
op3 = 0.3*LazyTensor(b_l, b_r, 3, subop3)
op3_ = 0.3*I1 ⊗ I2 ⊗ subop3

x1 = Ket(b_r, rand(ComplexF64, length(b_r)))
x2 = Ket(b_r, rand(ComplexF64, length(b_r)))
xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))
xbra2 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Addition
@test_throws ArgumentError op1 + op2
@test_throws ArgumentError op1 - op2
@test 1e-14 > D(-op1_, -op1)

# Test multiplication
@test_throws DimensionMismatch op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_)
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-12 > D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))
@test 1e-12 > D(op1_*dagger(0.3*op2), op1_*dagger(0.3*op2_))
@test 1e-12 > D(dagger(0.3*op2)*op1_, dagger(0.3*op2_)*op1_)
@test 1e-12 > D(dagger(0.3*op2)*op1, dagger(0.3*op2_)*op1_)


# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Test identityoperator
Idense = identityoperator(DenseOpType, b_r)
id = identityoperator(LazyTensor, b_r)
@test isa(id, LazyTensor)
@test dense(id) == Idense
@test 1e-11 > D(id*x1, x1)
@test id == identityoperator(LazyTensor, b1b) ⊗ identityoperator(LazyTensor, b2b) ⊗ identityoperator(LazyTensor, b3b)

Idense = identityoperator(DenseOpType, b_l)
id = identityoperator(LazyTensor, b_l)
@test isa(id, LazyTensor)
@test dense(id) == Idense
@test 1e-11 > D(xbra1*id, xbra1)
@test id == identityoperator(LazyTensor, b1a) ⊗ identityoperator(LazyTensor, b2a) ⊗ identityoperator(LazyTensor, b3a)


# Test tr and normalize
subop1 = randoperator(b1a)
I2 = dense(identityoperator(b2a))
subop3 = randoperator(b3a)
op = LazyTensor(b_l, b_l, [1, 3], (subop1, sparse(subop3)), 0.1)
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

@test tr(op) ≈ tr(op_)
op_normalized = normalize(op)
@test tr(op_) ≈ tr(op)
@test 1 ≈ tr(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test tr(op) != tr(op_copy)
@test 1 ≈ tr(op_copy)

# Test partial tr
subop1 = randoperator(b1a)
I2 = dense(identityoperator(b2a))
subop3 = randoperator(b3a)
op = LazyTensor(b_l, b_l, [1, 3], (subop1, sparse(subop3)), 0.1)
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

@test 1e-14 > D(ptrace(op_, 3), ptrace(op, 3))
@test 1e-14 > D(ptrace(op_, 2), ptrace(op, 2))
@test 1e-14 > D(ptrace(op_, 1), ptrace(op, 1))

@test 1e-14 > D(ptrace(op_, [2,3]), ptrace(op, [2,3]))
@test 1e-14 > D(ptrace(op_, [1,3]), ptrace(op, [1,3]))
@test 1e-14 > D(ptrace(op_, [1,2]), ptrace(op, [1,2]))

@test_throws ArgumentError ptrace(op, [1,2,3])

# Test expect
state = Ket(b_l, rand(ComplexF64, length(b_l)))
@test expect(op, state) ≈ expect(op_, state)

state = DenseOperator(b_l, b_l, rand(ComplexF64, length(b_l), length(b_l)))
@test expect(op, state) ≈ expect(op_, state)

# Permute systems
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I2 = dense(identityoperator(b2a, b2b))
op = LazyTensor(b_l, b_r, [1, 3], (subop1, sparse(subop3)))*0.1
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

@test 1e-14 > D(permutesystems(op, [1, 3, 2]), permutesystems(op_, [1, 3, 2]))
@test 1e-14 > D(permutesystems(op, [2, 1, 3]), permutesystems(op_, [2, 1, 3]))
@test 1e-14 > D(permutesystems(op, [2, 3, 1]), permutesystems(op_, [2, 3, 1]))
@test 1e-14 > D(permutesystems(op, [3, 1, 2]), permutesystems(op_, [3, 1, 2]))
@test 1e-14 > D(permutesystems(op, [3, 2, 1]), permutesystems(op_, [3, 2, 1]))


# Test gemv
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I2 = dense(identityoperator(b2a, b2b))
op = LazyTensor(b_l, b_r, [1, 3], (subop1, sparse(subop3)))*0.1
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

state = Ket(b_r, rand(ComplexF64, length(b_r)))
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm
b_l2 = GenericBasis(17)
b_r2 = GenericBasis(13)
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I2 = dense(identityoperator(b2a, b2b))
op = LazyTensor(b_l, b_r, [1, 3], (subop1, sparse(subop3)))*0.1
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

state = randoperator(b_r, b_r2)
result_ = randoperator(b_l, b_r2)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b_l2, b_l)
result_ = randoperator(b_l2, b_r)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

# Test calling gemv with non-complex factors
state = Ket(b_r, rand(ComplexF64, length(b_r)))
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state)
@test 1e-13 > D(result, op_*state)

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op)
@test 1e-13 > D(result, state*op_)

# Test gemm errors
test_op = test_lazytensor(b1a, b1a, rand(2, 2))
test_lazy = LazyTensor(tensor(b1a, b1a), [1, 2], (test_op, test_op))
test_ket = Ket(tensor(b1a, b1a), rand(4))

@test_throws ArgumentError QuantumOpticsBase.mul!(copy(test_ket),test_lazy,test_ket,alpha,beta)
@test_throws ArgumentError QuantumOpticsBase.mul!(copy(dagger(test_ket)),dagger(test_ket),test_lazy,alpha,beta)

# Test type stability of constructor
callT = typeof.((FockBasis(2) ⊗ FockBasis(2), 1, destroy(FockBasis(2))))
T = Core.Compiler.return_type(LazyTensor, callT)
@test all(map(isconcretetype, T.parameters))

end # testset
