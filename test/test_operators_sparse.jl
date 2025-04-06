using Test
using QuantumOpticsBase
import QuantumInterface: IncompatibleBases
using Random, SparseArrays, LinearAlgebra


# Custom operator type for testing error msg
mutable struct TestOperator{BL<:Basis,BR<:Basis} <: AbstractOperator; end

@testset "operators-sparse" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)
sprandop(b1, b2) = sparse(randoperator(b1, b2))
sprandop(b) = sprandop(b, b)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

# Test creation
@test_throws DimensionMismatch DenseOperator(b1a, spzeros(ComplexF64, 3, 2))
@test_throws DimensionMismatch DenseOperator(b1a, b1b, spzeros(ComplexF64, 3, 2))
op1 = SparseOperator(b1a, b1b, sparse([1 1 1; 1 1 1]))
op2 = sparse(DenseOperator(b1b, b1a, [1 1; 1 1; 1 1]))
@test op1 == dagger(op2)

# Test transpose
@test transpose(op2) == conj(op2')

# Test copy
op1 = sparse(randoperator(b1a))
op2 = copy(op1)
@test isequal(op1, op2)
@test op1 == op2
@test !(op1.data === op2.data)
op2.data[1,1] = complex(10.)
@test op1.data[1,1] != op2.data[1,1]

@test QuantumOpticsBase.is_const(op1)

# Arithmetic operations
# =====================
op_zero = SparseOperator(b_l, b_r)
op1 = sprandop(b_l, b_r)
op2 = sprandop(b_l, b_r)
op3 = sprandop(b_l, b_r)
op1_ = dense(op1)
op2_ = dense(op2)
op3_ = dense(op3)

x1 = randstate(b_r)
x2 = randstate(b_r)

xbra1 = dagger(randstate(b_l))
xbra2 = dagger(randstate(b_l))

# Addition
@test_throws IncompatibleBases op1 + dagger(op2)
@test 1e-14 > D(op1+op2, op1_+op2_)
@test 1e-14 > D(op1+op2, op1+op2_)
@test 1e-14 > D(op1+op2, op1_+op2)

# Subtraction
@test_throws IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1-op2, op1_-op2_)
@test 1e-14 > D(op1-op2, op1-op2_)
@test 1e-14 > D(op1-op2, op1_-op2)
@test 1e-14 > D(op1+(-op2), op1_ - op2_)
@test 1e-14 > D(op1+(-1*op2), op1_ - op2_)

# Test multiplication
@test_throws IncompatibleBases op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), (op1_+op2_)*(x1+0.3*x2))

@test 1e-11 > D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_)
@test 1e-11 > D(xbra1*op1 + 0.3*xbra2*op1, xbra1*op1_ + 0.3*xbra2*op1_)
@test 1e-11 > D((xbra1+0.3*xbra2)*(op1+op2), (xbra1+0.3*xbra2)*(op1_+op2_))

@test 1e-12 > D(op1*dagger(0.3*op2), op1_*dagger(0.3*op2_))
@test 1e-12 > D(0.3*dagger(op2*dagger(op1)), 0.3*dagger(op2_*dagger(op1_)))
@test 1e-12 > D((op1 + op2)*dagger(0.3*op3), (op1_ + op2_)*dagger(0.3*op3_))
@test 1e-12 > D(0.3*op1*dagger(op3) + 0.3*op2*dagger(op3), 0.3*op1_*dagger(op3_) + 0.3*op2_*dagger(op3_))

# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Conjugation
tmp = copy(op1)
conj!(tmp)
@test tmp == conj(op1) && conj(tmp.data) == op1.data

# Test identityoperator
Idense = identityoperator(DenseOpType, b_r)
I = identityoperator(SparseOpType, b_r)
@test isa(I, SparseOpType)
@test dense(I) == Idense
@test 1e-11 > D(I*x1, x1)
@test I == identityoperator(SparseOpType, b1b) ⊗ identityoperator(SparseOpType, b2b) ⊗ identityoperator(SparseOpType, b3b)

Idense = identityoperator(DenseOpType, b_l)
I = identityoperator(SparseOpType, b_l)
@test isa(I, SparseOpType)
@test dense(I) == Idense
@test 1e-11 > D(xbra1*I, xbra1)
@test I == identityoperator(SparseOpType, b1a) ⊗ identityoperator(SparseOpType, b2a) ⊗ identityoperator(SparseOpType, b3a)

IEye = identityoperator(b_l)
@test isa(IEye, EyeOpType)
@test sparse(IEye) == I
Icomp = identityoperator(b1a) ⊗ identityoperator(b2a) ⊗ identityoperator(b3a)
@test IEye == Icomp

for _IEye in (identityoperator(b_l), identityoperator(b1a, b1b))
    for IEye in (_IEye, -_IEye, _IEye', dagger(_IEye))
        @test isa(2*IEye, SparseOpType)
        @test isa(IEye*2, SparseOpType)
        @test isa(IEye/2, SparseOpType)
        @test isa(IEye+sparse(IEye), SparseOpType)
        @test isa(sparse(IEye)+IEye, SparseOpType)
        @test isa(sparse(IEye)-IEye, SparseOpType)
        @test isa(IEye-sparse(IEye), SparseOpType)
        @test isa(IEye+IEye, SparseOpType)
        @test isa(IEye-IEye, SparseOpType)
        @test isa(-IEye, SparseOpType)
        if VERSION.major == 1 && VERSION.minor == 6
            # julia 1.6 LTS, something's broken here
            @test_skip isa(tensor(IEye, sparse(IEye)), SparseOpType)
            @test_skip isa(tensor(sparse(IEye), IEye), SparseOpType)
            @test_skip isa(tensor(IEye, IEye), SparseOpType)
        else
            @test isa(tensor(IEye, sparse(IEye)), SparseOpType)
            @test isa(tensor(sparse(IEye), IEye), SparseOpType)
            @test isa(tensor(IEye, IEye), SparseOpType)
        end
    end
end

# Test tr and normalize
op = sparse(DenseOperator(GenericBasis(3), [1 3 2;5 2 2;-1 2 5]))
@test 8 == tr(op)
op_normalized = normalize(op)
@test 8 == tr(op)
@test 1 == tr(op_normalized)
# op_ = normalize!(op)
# @test op_ === op
# @test 1 == tr(op)

# Test partial tr
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
b_l = b1 ⊗ b2 ⊗ b3
op1 = sprandop(b1)
op2 = sprandop(b2)
op3 = sprandop(b3)
op123 = op1 ⊗ op2 ⊗ op3
op123_ = dense(op123)

@test 1e-14 > D(ptrace(op123_, 3), ptrace(op123, 3))
@test 1e-14 > D(ptrace(op123_, 2), ptrace(op123, 2))
@test 1e-14 > D(ptrace(op123_, 1), ptrace(op123, 1))

@test 1e-14 > D(ptrace(op123_, [2,3]), ptrace(op123, [2,3]))
@test 1e-14 > D(ptrace(op123_, [1,3]), ptrace(op123, [1,3]))
@test 1e-14 > D(ptrace(op123_, [1,2]), ptrace(op123, [1,2]))

@test_throws ArgumentError ptrace(op123, [1,2,3])

# Test expect
state = randstate(b_l)
@test expect(op123, state) ≈ expect(op123_, state)

state = randoperator(b_l)
@test expect(op123, state) ≈ expect(op123_, state)

@test_throws IncompatibleBases expect(op1, op2)

# Tensor product
# ==============
b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)
b_l = b1a ⊗ b2a ⊗ b3a
b_r = b1b ⊗ b2b ⊗ b3b
op1a = sprandop(b1a, b1b)
op1b = sprandop(b1a, b1b)
op2a = sprandop(b2a, b2b)
op2b = sprandop(b2a, b2b)
op3a = sprandop(b3a, b3b)
op1a_ = dense(op1a)
op1b_ = dense(op1b)
op2a_ = dense(op2a)
op2b_ = dense(op2b)
op3a_ = dense(op3a)
op123 = op1a ⊗ op2a ⊗ op3a
op123_ = op1a_ ⊗ op2a_ ⊗ op3a_
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
@test 1e-13 > D((op1a ⊗ op2a) ⊗ op3a, (op1a_ ⊗ op2a_) ⊗ op3a_)
@test 1e-13 > D(op1a ⊗ (op2a ⊗ op3a), op1a_ ⊗ (op2a_ ⊗ op3a_))
@test 1e-13 > D(op1a ⊗ (op2a ⊗ op3a), op1a_ ⊗ (op2a_ ⊗ op3a))

# Linearity
@test 1e-13 > D(op1a ⊗ (0.3*op2a), op1a_ ⊗ (0.3*op2a_))
@test 1e-13 > D(0.3*(op1a ⊗ op2a), 0.3*(op1a_ ⊗ op2a_))
@test 1e-13 > D((0.3*op1a) ⊗ op2a, (0.3*op1a_) ⊗ op2a_)
@test 1e-13 > D(0.3*(op1a ⊗ op2a), 0.3*(op1a_ ⊗ op2a_))
@test 1e-13 > D(0.3*(op1a ⊗ op2a), 0.3*(op1a ⊗ op2a_))

# Distributivity
@test 1e-13 > D(op1a ⊗ (op2a + op2b), op1a_ ⊗ (op2a_ + op2b_))
@test 1e-13 > D(op1a ⊗ op2a + op1a ⊗ op2b, op1a_ ⊗ op2a_ + op1a_ ⊗ op2b_)
@test 1e-13 > D((op2a + op2b) ⊗ op3a, (op2a_ + op2b_) ⊗ op3a_)
@test 1e-13 > D(op2a ⊗ op3a + op2b ⊗ op3a, op2a_ ⊗ op3a_ + op2b_ ⊗ op3a_)
@test 1e-13 > D(op2a ⊗ op3a + op2b ⊗ op3a, op2a ⊗ op3a_ + op2b_ ⊗ op3a_)

# Mixed-product property
@test 1e-13 > D((op1a ⊗ op2a) * dagger(op1b ⊗ op2b), (op1a_ ⊗ op2a_) * dagger(op1b_ ⊗ op2b_))
@test 1e-13 > D((op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)), (op1a_*dagger(op1b_)) ⊗ (op2a_*dagger(op2b_)))
@test 1e-13 > D((op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)), (op1a_*dagger(op1b)) ⊗ (op2a_*dagger(op2b_)))

# Transpose
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a_ ⊗ op2a_))
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a ⊗ op2a_))
@test 1e-13 > D(dagger(op1a) ⊗ dagger(op2a), dagger(op1a_) ⊗ dagger(op2a_))


# Permute systems
op1 = sprandop(b1a, b1b)
op2 = sprandop(b2a, b2b)
op3 = sprandop(b3a, b3b)
op123 = op1⊗op2⊗op3

op132 = op1⊗op3⊗op2
@test 1e-14 > D(permutesystems(op123, [1, 3, 2]), op132)

op213 = op2⊗op1⊗op3
@test 1e-14 > D(permutesystems(op123, [2, 1, 3]), op213)

op231 = op2⊗op3⊗op1
@test 1e-14 > D(permutesystems(op123, [2, 3, 1]), op231)

op312 = op3⊗op1⊗op2
@test 1e-14 > D(permutesystems(op123, [3, 1, 2]), op312)

op321 = op3⊗op2⊗op1
@test 1e-14 > D(permutesystems(op123, [3, 2, 1]), op321)

# Test diagonaloperator
b = GenericBasis(4)
I = identityoperator(b)

@test diagonaloperator(b, [1, 1, 1, 1]) == I
@test diagonaloperator(b, [1., 1., 1., 1.]) == I
@test diagonaloperator(b, [1im, 1im, 1im, 1im]) == 1im*I
@test diagonaloperator(b, [0:3;]) == sparse(DenseOperator(b, Diagonal([0:3;])))

# Test gemv
op = sprandop(b_l, b_r)
op_ = dense(op)
xket = randstate(b_l)
xbra = dagger(xket)

state = randstate(b_r)
result_ = randstate(b_l)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.0),complex(0.))
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

state = dagger(randstate(b_l))
result_ = dagger(randstate(b_r))
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.0),complex(0.))
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm small version
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)

op = sprandop(b1, b2)
op_ = dense(op)

state = randoperator(b2, b3)
result_ = randoperator(b1, b3)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b3, b1)
result_ = randoperator(b3, b2)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

state = randoperator(b1, b3)
result_ = randoperator(b2, b3)
result = deepcopy(result_)
result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op',state,alpha,beta) # gemm! with lazy adjoint sparse
@test 1e-12 > D(result, alpha*op_'*state + beta*result_)

state = randoperator(b1, b2)
result_ = randoperator(b1, b1)
result = deepcopy(result_)
result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op',alpha,beta) # gemm! with lazy adjoint sparse
@test 1e-12 > D(result, alpha*state*op_' + beta*result_)

# Test gemm big version
b1 = GenericBasis(50)
b2 = GenericBasis(60)
b3 = GenericBasis(55)

op = sprandop(b1, b2)
op_ = dense(op)

state = randoperator(b2, b3)
result_ = randoperator(b1, b3)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,op,state,complex(1.),complex(0.))
@test 1e-11 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,op,state,alpha,beta)
@test 1e-11 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b3, b1)
result_ = randoperator(b3, b2)
result = deepcopy(result_)
QuantumOpticsBase.mul!(result,state,op,complex(1.),complex(0.))
@test 1e-11 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op,alpha,beta)
@test 1e-11 > D(result, alpha*state*op_ + beta*result_)

state = randoperator(b1, b2)
result_ = randoperator(b1, b1)
result = deepcopy(result_)
result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
QuantumOpticsBase.mul!(result,state,op',alpha,beta) # gemm! with lazy adjoint sparse
@test 1e-11 > D(result, alpha*state*op_' + beta*result_)

# Test remaining uncovered code
@test_throws DimensionMismatch SparseOperator(b1, b2, zeros(10, 10))
dat = sprandop(b1, b1).data
@test SparseOperator(b1, dat) == SparseOperator(b1, Matrix{ComplexF64}(dat))

@test_throws ArgumentError sparse(TestOperator{Basis,Basis}())

@test 2*SparseOperator(b1, dat) == SparseOperator(b1, dat)*2
@test copy(op1) == deepcopy(op1)

# Test Hermitian
bspin = SpinBasis(1//2)
bnlevel = NLevelBasis(2)
@test ishermitian(SparseOperator(bspin, bspin, sparse([1.0 im; -im 2.0])))
@test !ishermitian(SparseOperator(bspin, bnlevel, sparse([1.0 im; -im 2.0])))

# Test broadcasting
@test_throws DimensionMismatch op1 .+ op2
@test op1 .+ op1 == op1 + op1
op1 .= DenseOperator(op1)
@test isa(op1, SparseOpType)
# @test isa(op1 .+ DenseOperator(op1), DenseOpType) # Broadcasting of sparse .+ dense matrix results in sparse
op3 = sprandop(FockBasis(1),FockBasis(2))
@test_throws IncompatibleBases op1 .+ op3
@test_throws IncompatibleBases op1 .= op3
op_ = copy(op1)
op_ .+= op1
@test op_ == 2*op1

# Dimension mismatches
b1, b2, b3 = NLevelBasis.((2,3,4))  # N is not a type parameter
@test_throws DimensionMismatch mul!(randstate(b1), sparse(randoperator(b2)), randstate(b3))
@test_throws DimensionMismatch mul!(randstate(b1)', randstate(b3)', sparse(randoperator(b2)))
@test_throws DimensionMismatch mul!(randoperator(b1), sparse(randoperator(b2)), randoperator(b3))
@test_throws DimensionMismatch mul!(randoperator(b1), randoperator(b3)', sparse(randoperator(b2)))

end # testset

@testset "State-operator tensor products, sparse" begin
    b = FockBasis(2) ⊗ SpinBasis(1//2) ⊗ GenericBasis(2)
    b1, b2, b3 = b[1], b[2], b[3]

    o1 = sparse(randoperator(b1))
    v1 = sparse(randstate(b1))
    p1 = sparse(projector(v1))
    o2 = sparse(randoperator(b2))
    v2 = sparse(randstate(b2))
    p2 = sparse(projector(v2))
    o3 = sparse(randoperator(b3))
    v3 = sparse(randstate(b3))
    p3 = sparse(projector(v3))

    res = ((o1 ⊗ v2) * (o1 ⊗ v2'))
    @test res isa SparseOpType
    @test res.data ≈ (o1^2 ⊗ p2).data

    res = ((o1 ⊗ v2') * (o1 ⊗ v2))
    @test res isa SparseOpType
    @test res.data ≈ (o1^2).data

    res = ((o1 ⊗ v2 ⊗ o3) * (o1 ⊗ v2' ⊗ o3))
    @test res isa SparseOpType
    @test res.data ≈ (o1^2 ⊗ p2 ⊗ o3^2).data

    res = ((v1 ⊗ o2 ⊗ o3) * (v1' ⊗ o2 ⊗ o3))
    @test res isa SparseOpType
    @test res.data ≈ (p1 ⊗ o2^2 ⊗ o3^2).data

    res = ((v1 ⊗ o2 ⊗ v3) * (v1' ⊗ o2 ⊗ v3'))
    @test res isa SparseOpType
    @test res.data ≈ (p1 ⊗ o2^2 ⊗ p3).data
end
