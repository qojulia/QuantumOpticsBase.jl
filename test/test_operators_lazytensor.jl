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
@test isequal(x, x_)
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

# Test eltype
for T in (Float32, Float64, ComplexF32, ComplexF64)
    I2_T = identityoperator(T, b2a, b2b)
    op1_T = randoperator(T, b1a, b1b)
    op3_T = randoperator(T, b3a, b3b)
    x_T =  LazyTensor(b_l, b_r, [1, 3], (op1_T, sparse(op3_T)), T(0.3))
    @test eltype(x_T) == T
    @test eltype(dense(x_T)) == T
    @test eltype(sparse(x_T)) == T
end

# Test suboperators
@test QuantumOpticsBase.suboperator(x, 1) == op1
@test QuantumOpticsBase.suboperator(x, 3) == sparse(op3)
@test QuantumOpticsBase.suboperators(x, [1, 3]) == [op1, sparse(op3)]

# Test embed
x_1 = LazyTensor(b_l, b_r, [1], (op1,), 0.3)
x_1_sub = LazyTensor(b1a⊗b2a, b1b⊗b2b, [1], (op1,), 0.3)
@test embed(b_l, b_r, Dict([1,2]=>x_1_sub)) == x_1
@test embed(b_l, b_r, [1,2], x_1_sub) == x_1

x_12 = LazyTensor(b1a⊗b3a, b1b⊗b3b, [1,2], (op1, sparse(op3)), 0.3)
@test embed(b_l, b_r, Dict([1,3]=>x_12)) == x
@test embed(b_l, b_r, [1,3], x_12) == x


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

# Allowed addition
fac = randn()
@test dense(op3 + fac * op3) ≈ dense(op3) * (1 + fac)
@test dense(op3 - fac * op3) ≈ dense(op3) * (1 - fac)

# Forbidden addition
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


# Test gemv, mixing precisions
subop1 = randoperator(ComplexF32, b1a, b1b)
subop2 = randoperator(b2b, b2a)'  # test adjoint explicitly
subop3 = randoperator(b3b, b3a)'  # test adjoint explicitly
op = LazyTensor(b_l, b_r, [1, 2, 3], (subop1, subop2, subop3))*0.1
op_sp = LazyTensor(b_l, b_r, [1, 2, 3], sparse.((subop1, subop2, subop3)))*0.1
op_ = 0.1*subop1 ⊗ subop2 ⊗ subop3

state = Ket(b_r, rand(ComplexF32, length(b_r)))
state_sp = sparse(state)  # to test no-cache path
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-6 > D(result, op_*state)

QuantumOpticsBase.mul!(result,op,state_sp,complex(1.),complex(0.))
@test 1e-6 > D(result, op_*state)

QuantumOpticsBase.mul!(result,op_sp,state,complex(1.),complex(0.))
@test 1e-6 > D(result, op_*state)

@test lazytensor_cachesize() > 0  # the cache should have some entries by now

lazytensor_disable_cache()
@test !QuantumOpticsBase.lazytensor_use_cache()

QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-6 > D(result, op_*state)

lazytensor_enable_cache(; maxsize=8)  # tiny cache that won't hold anything
@test QuantumOpticsBase.lazytensor_use_cache()

@test lazytensor_cachesize() <= 8

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-6 > D(result, alpha*op_*state + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op_sp,state,alpha,beta)
@test 1e-6 > D(result, alpha*op_*state + beta*result_)

lazytensor_clear_cache()
lazytensor_enable_cache(; maxsize=2^30, maxrelsize=2^30 / Sys.total_memory())
lazytensor_enable_cache()

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op_sp,complex(1.),complex(0.))
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op_sp,alpha,beta)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm
b_l2 = GenericBasis(17)
b_r2 = GenericBasis(13)
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
op = LazyTensor(b_l, b_r, [1, 2, 3], (subop1, subop2, sparse(subop3)))*0.1
op_sp = LazyTensor(b_l, b_r, [1, 2, 3], sparse.((subop1, subop2, subop3)))*0.1
op_ = 0.1*subop1 ⊗ subop2 ⊗ subop3

state = randoperator(b_r, b_r2)
result_ = randoperator(b_l, b_r2)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op_sp,state,complex(1.),complex(0.))
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op_sp,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b_l2, b_l)
result_ = randoperator(b_l2, b_r)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op_sp,complex(1.),complex(0.))
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op_sp,alpha,beta)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

# Test calling gemv with non-complex factors
state = Ket(b_r, rand(ComplexF64, length(b_r)))
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state)
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op_sp,state)
@test 1e-13 > D(result, op_*state)

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op)
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op_sp)
@test 1e-13 > D(result, state*op_)

# Test scaled isometry action
op = LazyTensor(b_l, b_r, (), (), 0.5)
op_ = sparse(op)
state = randoperator(b_r, b_r2)
result_ = randoperator(b_l, b_r2)
alpha = complex(1.5)
beta = complex(2.1)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

# Test single operator, no isometries
subop1 = randoperator(b1a, b1a)
op = LazyTensor(b_l, b_l, [1], (subop1,), 0.5)
op_sp = LazyTensor(b_l, b_l, [1], (sparse(subop1),), 0.5)
op_ = sparse(op)
state = randoperator(b_l, b_r2)
result_ = randoperator(b_l, b_r2)
alpha = complex(1.5)
beta = complex(2.1)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op_sp,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

# Test scaled identity
op = LazyTensor(b_l, b_l, (), (), 0.5)
op_ = sparse(op)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

# Test gemm errors
test_op = test_lazytensor(b1a, b1a, rand(2, 2))
test_lazy = LazyTensor(tensor(b1a, b1a), [1, 2], (test_op, test_op))
test_ket = Ket(tensor(b1a, b1a), rand(4))

@test_throws MethodError QuantumOpticsBase.mul!(copy(test_ket),test_lazy,test_ket,alpha,beta)
@test_throws MethodError QuantumOpticsBase.mul!(copy(dagger(test_ket)),dagger(test_ket),test_lazy,alpha,beta)

# Test type stability of constructor
callT = typeof((FockBasis(2) ⊗ FockBasis(2), 1, destroy(FockBasis(2))))
T = Core.Compiler.return_type(LazyTensor, callT)
@test all(map(isconcretetype, T.parameters))

# Test mul! of adjoint dense operator with sparse lazy tensor
## adjoint from the left
lop = LazyTensor(b1a⊗b1b, b2a⊗b2b, 1, SparseOperator(randoperator(b1a,b2a)))
dop = randoperator(b1a⊗b1b, b3a⊗b3b)
@test dop'*lop ≈ dop'*lop ≈ Operator(dop.basis_r, lop.basis_r, dop.data'*dense(lop).data)
@test lop'*dop ≈ Operator(lop.basis_r, dop.basis_r, dense(lop).data'*dop.data)
## adjoint from the right
lop = LazyTensor(b1a⊗b1b, b2a⊗b2b, 1, SparseOperator(randoperator(b1a,b2a)))
dop = randoperator(b3a⊗b3b, b2a⊗b2b)
@test dop*lop' ≈ Operator(dop.basis_l, lop.basis_l, dop.data*dense(lop).data')
@test lop*dop' ≈ Operator(lop.basis_l, dop.basis_l, dense(lop).data*dop.data')

# Dimension mismatches for LazyTensor with sparse
b1, b2 = NLevelBasis.((2, 3))
Lop1 = LazyTensor(b1^2, b2^2, 2, sparse(randoperator(b1, b2)))
@test_throws DimensionMismatch Lop1*Lop1
@test_throws DimensionMismatch dense(Lop1)*Lop1
@test_throws DimensionMismatch sparse(Lop1)*Lop1
@test_throws DimensionMismatch Lop1*dense(Lop1)
@test_throws DimensionMismatch Lop1*sparse(Lop1)

end # testset

@testset "LazyTensor: explicit isometries" begin

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

# Test explicit identities and isometries
bl = FockBasis(2) ⊗ GenericBasis(2) ⊗ SpinBasis(1//2) ⊗ GenericBasis(1) ⊗ GenericBasis(2)
br = FockBasis(2) ⊗ GenericBasis(2) ⊗ SpinBasis(1//2) ⊗ GenericBasis(2) ⊗ GenericBasis(1)

iso = identityoperator(bl.bases[5], br.bases[5])

n1 = LazyTensor(bl, br, (1,3), (number(bl.bases[1]), sigmax(bl.bases[3])))
n1_sp = LazyTensor(bl, br, (1,2,3,5), (number(bl.bases[1]), identityoperator(bl.bases[2]), sigmax(bl.bases[3]), iso))
n1_de = LazyTensor(bl, br, (1,2,3,5), (dense(number(bl.bases[1])), identityoperator(bl.bases[2]), sigmax(bl.bases[3]), iso))

@test dense(n1) == dense(n1_sp)
@test dense(n1) == dense(n1_de)

state = randoperator(br,br)

@test 1e-12 > D(n1 * state, n1_sp * state)
@test 1e-12 > D(n1 * state, n1_de * state)

out = randoperator(bl,br)
alpha = randn()
beta = randn()
out_ref = mul!(copy(out), n1, state, alpha, beta)
@test 1e-12 > D(out_ref, mul!(copy(out), n1_sp, state, alpha, beta))
@test 1e-12 > D(out_ref, mul!(copy(out), n1_de, state, alpha, beta))

out_NaN = NaN * out 
out_ref = mul!(copy(out_NaN), n1, state, alpha, 0)
@test 1e-12 > D(out_ref, mul!(copy(out_NaN), n1_sp, state, alpha, 0))
@test 1e-12 > D(out_ref, mul!(copy(out_NaN), n1_de, state, alpha, 0))

out_ref = mul!(copy(out), n1, state, 0, beta)
@test 1e-12 > D(out_ref, mul!(copy(out), n1_sp, state, 0, beta))
@test 1e-12 > D(out_ref, mul!(copy(out), n1_de, state, 0, beta))

out_NaN = NaN * out 
out_ref = mul!(copy(out_NaN), n1, state, 0, 0)
@test 1e-12 > D(out_ref, mul!(copy(out_NaN), n1_sp, state, 0, 0))
@test 1e-12 > D(out_ref, mul!(copy(out_NaN), n1_de, state, 0, 0))

state = randoperator(bl,bl)
out_ref = mul!(copy(out), state, n1, alpha, beta)
@test 1e-12 > D(out_ref, mul!(copy(out), state, n1_sp, alpha, beta))
@test 1e-12 > D(out_ref, mul!(copy(out), state, n1_de, alpha, beta))

end
