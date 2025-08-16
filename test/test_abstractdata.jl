@testitem "test_abstractdata" begin
using QuantumOpticsBase
using Test
using LinearAlgebra
import LinearAlgebra: mul!
using Random
import Base: ==, -

# Implement custom type with AbstractArray interface
mutable struct TestData{T,N,X} <: AbstractArray{T,N}
    x::X
    function TestData(x::X) where X
        x_ = convert(Matrix{ComplexF64}, x)
        new{ComplexF64,length(axes(x)),typeof(x_)}(x_)
    end
end
Base.size(A::TestData) = size(A.x)
Base.getindex(A::TestData, inds...) = getindex(A.x, inds...)
Base.setindex!(A::TestData, val, inds...) = setindex!(A.x, val, inds...)
Base.isapprox(A::TestData,B::TestData,args...;kwargs...) = isapprox(A.x,B.x,args...;kwargs...)

# Additional methods
Base.copy(A::TestData) = TestData(copy(A.x))
Base.kron(A::TestData,B::TestData) = TestData(kron(A.x,B.x))
# -(A::TestData) = TestData(-A.x)

# Auxiliary functions
function randtestoperator(args...)
    op_ = randoperator(args...)
    return Operator(op_.basis_l,op_.basis_r,TestData(copy(op_.data)))
end

D(a::DataOperator,b::DataOperator,tol=1e-14) = isapprox(a.data,b.data,atol=tol)
D(a::StateVector,b::StateVector,tol=1e-14) = isapprox(a.data,b.data,atol=tol)
D(a::AbstractOperator,b::AbstractOperator,tol=1e-12) = (tol > abs(tracedistance_nh(dense(a), dense(b))))

@testset "abstract-data" begin

Random.seed!(0)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

# Test creation
@test_throws DimensionMismatch Operator(b1a, TestData([1 1 1; 1 1 1]))
@test_throws DimensionMismatch Operator(b1a, b1b, TestData([1 1; 1 1; 1 1]))
op1 = Operator(b1a, b1b, TestData([1 1 1; 1 1 1]))
op2 = Operator(b1b, b1a, TestData([1 1; 1 1; 1 1]))
op3 = op1*op2
@test op1 == dagger(op2)

# Test ' shorthand
@test dagger(op2) == op2'
@test transpose(op2) == conj(op2')

# Test copy
op1 = randtestoperator(b1a)
op2 = copy(op1)
@test op1.data == op2.data
@test !(op1.data === op2.data)
op2.data[1,1] = complex(10.)
@test op1.data[1,1] != op2.data[1,1]


# Arithmetic operations
# =====================
op_zero = DenseOperator(b_l, b_r)
op1 = randtestoperator(b_l, b_r)
op2 = randtestoperator(b_l, b_r)
op3 = randtestoperator(b_l, b_r)

x1 = Ket(b_r, rand(ComplexF64, length(b_r)))
x2 = Ket(b_r, rand(ComplexF64, length(b_r)))

xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))
xbra2 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Addition
@test_throws DimensionMismatch op1 + dagger(op2)
@test D(op1 + op_zero, op1)
@test D(op1 + op2, op2 + op1)
@test D(op1 + (op2 + op3), (op1 + op2) + op3)

# Subtraction
@test_throws DimensionMismatch op1 - dagger(op2)
@test D(op1-op_zero, op1)
@test D(op1-op2, op1 + (-op2))
@test D(op1-op2, op1 + (-1*op2))
@test D(op1-op2-op3, op1-(op2+op3))

# Test multiplication
@test_throws DimensionMismatch op1*op2
@test D(op1*(x1 + 0.3*x2), op1*x1 + 0.3*op1*x2, 1e-12)
@test D((op1+op2)*(x1+0.3*x2), op1*x1 + 0.3*op1*x2 + op2*x1 + 0.3*op2*x2, 1e-12)

@test D((xbra1+0.3*xbra2)*op1, xbra1*op1 + 0.3*xbra2*op1)
@test D((xbra1+0.3*xbra2)*(op1+op2), xbra1*op1 + 0.3*xbra2*op1 + xbra1*op2 + 0.3*xbra2*op2)

@test D(op1*dagger(0.3*op2), 0.3*dagger(op2*dagger(op1)))
@test D((op1 + op2)*dagger(0.3*op3), 0.3*op1*dagger(op3) + 0.3*op2*dagger(op3), 1e-12)
@test D(0.3*(op1*dagger(op2)), op1*(0.3*dagger(op2)))

tmp = copy(op1)
conj!(tmp)
@test tmp == conj(op1) && conj(tmp.data) == op1.data

# Internal layout
b1 = GenericBasis(2)
b2 = GenericBasis(3)
b3 = GenericBasis(4)
op1 = randtestoperator(b1, b2)
op2 = randtestoperator(b2, b3)
x1 = randstate(b2)
d1 = op1.data
d2 = op2.data
v = x1.data
@test (op1*x1).data ≈ [d1[1,1]*v[1] + d1[1,2]*v[2] + d1[1,3]*v[3], d1[2,1]*v[1] + d1[2,2]*v[2] + d1[2,3]*v[3]]
@test (op1*op2).data[2,3] ≈ d1[2,1]*d2[1,3] + d1[2,2]*d2[2,3] + d1[2,3]*d2[3,3]

# Test division
@test D(op1/7, (1/7)*op1)

# Tensor product
# ==============
op1a = randtestoperator(b1a, b1b)
op1b = randtestoperator(b1a, b1b)
op2a = randtestoperator(b2a, b2b)
op2b = randtestoperator(b2a, b2b)
op3a = randtestoperator(b3a, b3b)
op123 = op1a ⊗ op2a ⊗ op3a
@test isa(op123.data,TestData)
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
@test D((op1a ⊗ op2a) ⊗ op3a, op1a ⊗ (op2a ⊗ op3a))

# Linearity
@test D(op1a ⊗ (0.3*op2a), 0.3*(op1a ⊗ op2a))
@test D((0.3*op1a) ⊗ op2a, 0.3*(op1a ⊗ op2a))

# Distributivity
@test D(op1a ⊗ (op2a + op2b), op1a ⊗ op2a + op1a ⊗ op2b)
@test D((op2a + op2b) ⊗ op3a, op2a ⊗ op3a + op2b ⊗ op3a)

# Mixed-product property
@test D((op1a ⊗ op2a) * dagger(op1b ⊗ op2b), (op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)))

# Transpose
@test D(dagger(op1a ⊗ op2a), dagger(op1a) ⊗ dagger(op2a))
@test D(dagger(op1a ⊗ op2a), dagger(op1a) ⊗ dagger(op2a))

# Internal layout
a = Ket(b1a, rand(ComplexF64, length(b1a)))
b = Ket(b2b, rand(ComplexF64, length(b2b)))
ab = a ⊗ dagger(b)
@test ab.data[2,3] == a.data[2]*conj(b.data[3])
@test ab.data[2,1] == a.data[2]*conj(b.data[1])

shape = tuple(op123.basis_l.shape..., op123.basis_r.shape...)
idx = LinearIndices(shape)[2, 1, 1, 3, 4, 5]
@test op123.data[idx] == op1a.data[2,3]*op2a.data[1,4]*op3a.data[1,5]
@test reshape(op123.data, shape...)[2, 1, 1, 3, 4, 5] == op1a.data[2,3]*op2a.data[1,4]*op3a.data[1,5]

idx = LinearIndices(shape)[2, 1, 1, 1, 3, 4]
@test op123.data[idx] == op1a.data[2,1]*op2a.data[1,3]*op3a.data[1,4]
@test reshape(op123.data, shape...)[2, 1, 1, 1, 3, 4] == op1a.data[2,1]*op2a.data[1,3]*op3a.data[1,4]


# Test tr and normalize
op = Operator(GenericBasis(3), TestData([1 3 2;5 2 2;-1 2 5]))
@test 8 == tr(op)
op_normalized = normalize(op)
@test 8 == tr(op)
@test 1 == tr(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test tr(op) != tr(op_copy)
@test 1 ≈ tr(op_copy)
@test op === normalize!(op)

# Test partial tr of operators
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
op1 = randtestoperator(b1)
op2 = randtestoperator(b2)
op3 = randtestoperator(b3)
op123 = op1 ⊗ op2 ⊗ op3

@test D(op1⊗op2*tr(op3), ptrace(op123, 3))
@test D(op1⊗op3*tr(op2), ptrace(op123, 2))
@test D(op2⊗op3*tr(op1), ptrace(op123, 1))

@test D(op1*tr(op2)*tr(op3), ptrace(op123, [2,3]))
@test D(op2*tr(op1)*tr(op3), ptrace(op123, [1,3]))
@test D(op3*tr(op1)*tr(op2), ptrace(op123, [1,2]))

@test_throws ArgumentError ptrace(op123, [1,2,3])
x = randtestoperator(b1, b1⊗b2)
@test_throws ArgumentError ptrace(x, [1])
x = randtestoperator(b1⊗b1⊗b2, b1⊗b2)
@test_throws ArgumentError ptrace(x, [1, 2])
x = randtestoperator(b1⊗b2)
@test_throws ArgumentError ptrace(x, [1, 2])
x = randtestoperator(b1⊗b2, b2⊗b1)
@test_throws ArgumentError ptrace(x, [1])

op1 = randtestoperator(b1, b2)
op2 = randtestoperator(b3)

@test D(op1*tr(op2), ptrace(op1⊗op2, 2))

# Test expect
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
op1 = randtestoperator(b1)
op2 = randtestoperator(b2)
op3 = randtestoperator(b3)
op123 = op1 ⊗ op2 ⊗ op3
b_l = b1 ⊗ b2 ⊗ b3

state = randstate(b_l)
@test expect(op123, state) ≈ dagger(state)*op123*state
@test expect(1, op1, state) ≈ expect(op1, ptrace(state, [2, 3]))
@test expect(2, op2, state) ≈ expect(op2, ptrace(state, [1, 3]))
@test expect(3, op3, state) ≈ expect(op3, ptrace(state, [1, 2]))

state = randtestoperator(b_l)
@test expect(op123, state) ≈ tr(op123*state)
@test expect(1, op1, state) ≈ expect(op1, ptrace(state, [2, 3]))
@test expect(2, op2, state) ≈ expect(op2, ptrace(state, [1, 3]))
@test expect(3, op3, state) ≈ expect(op3, ptrace(state, [1, 2]))

# Permute systems
op1 = randtestoperator(b1a, b1b)
op2 = randtestoperator(b2a, b2b)
op3 = randtestoperator(b3a, b3b)
op123 = op1⊗op2⊗op3

op132 = op1⊗op3⊗op2
@test D(permutesystems(op123, [1, 3, 2]), op132)

op213 = op2⊗op1⊗op3
@test D(permutesystems(op123, [2, 1, 3]), op213)

op231 = op2⊗op3⊗op1
@test D(permutesystems(op123, [2, 3, 1]), op231)

op312 = op3⊗op1⊗op2
@test D(permutesystems(op123, [3, 1, 2]), op312)

op321 = op3⊗op2⊗op1
@test D(permutesystems(op123, [3, 2, 1]), op321)


# Test projector
xket = normalize(Ket(b_l, rand(ComplexF64, length(b_l))))
yket = normalize(Ket(b_l, rand(ComplexF64, length(b_l))))
xbra = dagger(xket)
ybra = dagger(yket)

@test D(projector(xket)*xket, xket)
@test D(xbra*projector(xket), xbra)
@test D(projector(xbra)*xket, xket)
@test D(xbra*projector(xbra), xbra)
@test D(ybra*projector(yket, xbra), xbra)
@test D(projector(yket, xbra)*xket, yket)

# Test gemv
b1 = GenericBasis(3)
b2 = GenericBasis(5)
op = randtestoperator(b1, b2)
xket = randstate(b2)
xbra = dagger(randstate(b1))
rket = randstate(b1)
rbra = dagger(randstate(b2))
alpha = complex(0.7, 1.5)
beta = complex(0.3, 2.1)
@test isa(op.data, TestData)

rket_ = deepcopy(rket)
QuantumOpticsBase.mul!(rket_,op,xket,complex(1.0),complex(0.))
@test D(rket_, op*xket)

@test isa(op.data, TestData)
rket_ = deepcopy(rket)
QuantumOpticsBase.mul!(rket_,op,xket,alpha,beta)
@test D(rket_, alpha*op*xket + beta*rket)

rbra_ = deepcopy(rbra)
QuantumOpticsBase.mul!(rbra_,xbra,op,complex(1.0),complex(0.))
@test D(rbra_, xbra*op)

rbra_ = deepcopy(rbra)
QuantumOpticsBase.mul!(rbra_,xbra,op,alpha,beta)
@test D(rbra_, alpha*xbra*op + beta*rbra)

# Test gemm
b1 = GenericBasis(37)
b2 = GenericBasis(53)
b3 = GenericBasis(41)
op1 = randtestoperator(b1, b2)
op2 = randtestoperator(b2, b3)
r = randtestoperator(b1, b3)
alpha = complex(0.7, 1.5)
beta = complex(0.3, 2.1)

r_ = deepcopy(r)
QuantumOpticsBase.mul!(r_,op1,op2,complex(1.0),complex(0.))
@test D(r_, op1*op2)

r_ = deepcopy(r)
QuantumOpticsBase.mul!(r_,op1,op2,alpha,beta)
@test D(r_, alpha*op1*op2 + beta*r, 1e-12)

# Test Hermitian
bspin = SpinBasis(1//2)
bnlevel = NLevelBasis(2)
@test ishermitian(Operator(bspin, bspin, TestData([1.0 im; -im 2.0]))) == true
@test ishermitian(Operator(bspin, bnlevel, TestData([1.0 im; -im 2.0]))) == false

# Test broadcasting
op1_ = copy(op1)
@test isa(op1_.data, TestData)
op1 .= 2*op1
@test op1 == op1_ .+ op1_
op1 .= op1_
@test op1 == op1_
op1 .= op1_ .+ 3 * op1_
@test op1 == 4*op1_
@test_throws DimensionMismatch op1 .= op2
bf = FockBasis(3)
op3 = randtestoperator(bf)
@test_throws QuantumOpticsBase.IncompatibleBases op1 .+ op3

####################
# Test lazy tensor #
b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(6)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

op1 = randtestoperator(b1a, b1b)
op2 = randtestoperator(b2a, b2b)
op3 = randtestoperator(b3a, b3b)

# Test creation
@test_throws AssertionError LazyTensor(b_l, b_r, [1], (randtestoperator(b1a),))
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], (op1,))
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], (op1, sparse(randtestoperator(b_l, b_l))))
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], (randtestoperator(b_r, b_r), sparse(op2)))

# @test LazyTensor(b_l, b_r, [2, 1], [op2, op1]) == LazyTensor(b_l, b_r, [1, 2], [op1, op2])
x = randtestoperator(b2a)
@test LazyTensor(b_l, 2, x) == LazyTensor(b_l, b_l, [2], (x,))

# Test copy
x = 2*LazyTensor(b_l, b_r, [1,2], (randtestoperator(b1a, b1b), sparse(randtestoperator(b2a, b2b))))
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
@test D(0.3*op1⊗dense(I2)⊗op3, dense(x))
@test D(0.3*sparse(op1)⊗I2⊗sparse(op3), sparse(x))

# Test suboperators
@test QuantumOpticsBase.suboperator(x, 1) == op1
@test QuantumOpticsBase.suboperator(x, 3) == sparse(op3)
@test QuantumOpticsBase.suboperators(x, [1, 3]) == [op1, sparse(op3)]


# Arithmetic operations
subop1 = randtestoperator(b1a, b1b)
subop2 = randtestoperator(b2a, b2b)
subop3 = randtestoperator(b3a, b3b)
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
@test D(-op1_, -op1, 1e-12)

# Test multiplication
@test_throws DimensionMismatch op1*op2
@test D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_)
@test D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))
@test D(op1_*dagger(0.3*op2), op1_*dagger(0.3*op2_))
@test D(dagger(0.3*op2)*op1_, dagger(0.3*op2_)*op1_)
@test D(dagger(0.3*op2)*op1, dagger(0.3*op2_)*op1_)


#####################
# Test lazy product #
op1a = randtestoperator(b_l, b_r)
op1b = randtestoperator(b_r, b_l)
op2a = randtestoperator(b_l, b_r)
op2b = randtestoperator(b_r, b_l)
op3a = randtestoperator(b_l, b_l)
op1 = LazyProduct([op1a, sparse(op1b)])*0.1
op1_ = 0.1*(op1a*op1b)
op2 = LazyProduct([sparse(op2a), op2b], 0.3)
op2_ = 0.3*(op2a*op2b)
op3 = LazyProduct(op3a)
op3_ = op3a

x1 = Ket(b_l, rand(ComplexF64, length(b_l)))
x2 = Ket(b_l, rand(ComplexF64, length(b_l)))
xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))
xbra2 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Addition
@test D(2.1*op1 + 0.3*op2, 2.1*op1_+0.3*op2_)
@test D(-op1_, -op1)

# Test multiplication
@test_throws DimensionMismatch op1a*op1a
@test D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2), 1e-11)
@test D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_, 1e-11)
@test D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2, 1e-11)
@test D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_), 1e-11)
@test 0.3*LazyProduct(op1, sparse(op2)) == LazyProduct([op1, sparse(op2)], 0.3)
@test 0.3*LazyProduct(op1)*LazyProduct(sparse(op2)) == LazyProduct([op1, sparse(op2)], 0.3)


# Test gemv
op1 = randtestoperator(b_l, b_r)
op2 = randtestoperator(b_r, b_l)
op3 = randtestoperator(b_l, b_r)
op = LazyProduct([op1, sparse(op2), op3], 0.2)
op_ = 0.2*op1*op2*op3
tmp = 0.2*prod(dense.([op1*op2*op3]))

state = Ket(b_r, rand(ComplexF64, length(b_r)))
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = copy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test D(result, op_*state, 1e-11)

result = copy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test D(result, alpha*op_*state + beta*result_, 1e-9)

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = copy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test D(result, state*op_, 1e-9)

result = copy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test D(result, alpha*state*op_ + beta*result_, 1e-9)

# Test gemm
op1 = randtestoperator(b_l, b_r)
op2 = randtestoperator(b_r, b_l)
op3 = randtestoperator(b_l, b_r)
op = LazyProduct([op1, sparse(op2), op3], 0.2)
op_ = 0.2*op1*op2*op3

state = randtestoperator(b_r, b_r)
result_ = randtestoperator(b_l, b_r)
result = copy(result_)
result.data[1] = 1
result.data[1] != result_.data[1] || error("")
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test D(result, op_*state, 1e-9)

result = copy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test D(result, alpha*op_*state + beta*result_, 1e-9)

state = randtestoperator(b_l, b_l)
result_ = randtestoperator(b_l, b_r)
result = copy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test D(result, state*op_, 1e-9)

result = copy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test D(result, alpha*state*op_ + beta*result_, 1e-9)

end # testset
end
