using Test
using QuantumOpticsBase
using Random, SparseArrays, LinearAlgebra

@testset "operators-dense" begin

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
@test_throws DimensionMismatch DenseOperator(b1a, [1 1 1; 1 1 1])
@test_throws DimensionMismatch DenseOperator(b1a, b1b, [1 1; 1 1; 1 1])
op1 = DenseOperator(b1a, b1b, [1 1 1; 1 1 1])
op2 = DenseOperator(b1b, b1a, [1 1; 1 1; 1 1])
@test op1 == dagger(op2)

## Stacking Kets to make an Operator
### signle basis
ψlist = basisstate.([GenericBasis(4)], 1:2)
@test Operator(ψlist...) == Operator(ψlist) == Operator(ψlist[1].basis, GenericBasis(length(ψlist)), hcat(getfield.(ψlist,:data)...))
@test Operator(FockBasis(length(ψlist)-1), ψlist...) == Operator(FockBasis(length(ψlist)-1), ψlist) == Operator(ψlist[1].basis, FockBasis(length(ψlist)-1),  hcat(getfield.(ψlist,:data)...))
@test Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1), ψlist...) == Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1), ψlist) == Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1),  hcat(getfield.(ψlist,:data)...))
ψlist = vcat(ψlist, basisstate.(Real, [NLevelBasis(prod(ψlist[1].basis.shape))], [1,prod(ψlist[1].basis.shape)]))
@test Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1), ψlist...) == Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1), ψlist) == Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1),  hcat(getfield.(ψlist,:data)...))
### composite basis
ψlist = basisstate.([GenericBasis(4)^2], 1:2)
@test Operator(ψlist...) == Operator(ψlist) == Operator(ψlist[1].basis, GenericBasis(length(ψlist)), hcat(getfield.(ψlist,:data)...))
@test Operator(FockBasis(length(ψlist)-1), ψlist...) == Operator(FockBasis(length(ψlist)-1), ψlist) == Operator(ψlist[1].basis, FockBasis(length(ψlist)-1),  hcat(getfield.(ψlist,:data)...))
@test Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1), ψlist...) == Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1), ψlist) == Operator(NLevelBasis(prod(ψlist[1].basis.shape)), FockBasis(length(ψlist)-1),  hcat(getfield.(ψlist,:data)...))
ψlist2= vcat(ψlist, basisstate.(Float64, [NLevelBasis(prod(ψlist[1].basis.shape))], range(prod(ψlist[1].basis.shape);step=-1,length=length(ψlist))))
@test Operator(NLevelBasis(prod(ψlist2[1].basis.shape)), FockBasis(length(ψlist)-1)^2, ψlist2...) == Operator(NLevelBasis(prod(ψlist2[1].basis.shape)), FockBasis(length(ψlist)-1)^2, ψlist2) == Operator(NLevelBasis(prod(ψlist2[1].basis.shape)), FockBasis(length(ψlist)-1)^2,  hcat(getfield.(ψlist2,:data)...))

# Test ' shorthand
@test dagger(op2) == op2'
@test transpose(op2) == conj(op2')

# Test copy
op1 = randoperator(b1a)
op2 = copy(op1)
@test op1 == op2
@test isequal(op1, op2)
@test op1.data == op2.data
@test !(op1.data === op2.data)
op2.data[1,1] = complex(10.)
@test op1.data[1,1] != op2.data[1,1]


# Arithmetic operations
# =====================
op_zero = DenseOperator(b_l, b_r)
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_l, b_r)
op3 = randoperator(b_l, b_r)

x1 = Ket(b_r, rand(ComplexF64, length(b_r)))
x2 = Ket(b_r, rand(ComplexF64, length(b_r)))

xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))
xbra2 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Addition
@test_throws DimensionMismatch op1 + dagger(op2)
@test 1e-14 > D(op1 + op_zero, op1)
@test 1e-14 > D(op1 + op2, op2 + op1)
@test 1e-14 > D(op1 + (op2 + op3), (op1 + op2) + op3)

# Subtraction
@test_throws DimensionMismatch op1 - dagger(op2)
@test 1e-14 > D(op1-op_zero, op1)
@test 1e-14 > D(op1-op2, op1 + (-op2))
@test 1e-14 > D(op1-op2, op1 + (-1*op2))
@test 1e-14 > D(op1-op2-op3, op1-(op2+op3))

# Test multiplication
@test_throws DimensionMismatch op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1*x1 + 0.3*op1*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), op1*x1 + 0.3*op1*x2 + op2*x1 + 0.3*op2*x2)

@test 1e-11 > D((xbra1+0.3*xbra2)*op1, xbra1*op1 + 0.3*xbra2*op1)
@test 1e-11 > D((xbra1+0.3*xbra2)*(op1+op2), xbra1*op1 + 0.3*xbra2*op1 + xbra1*op2 + 0.3*xbra2*op2)

@test 1e-12 > D(op1*dagger(0.3*op2), 0.3*dagger(op2*dagger(op1)))
@test 1e-12 > D((op1 + op2)*dagger(0.3*op3), 0.3*op1*dagger(op3) + 0.3*op2*dagger(op3))
@test 1e-12 > D(0.3*(op1*dagger(op2)), op1*(0.3*dagger(op2)))

tmp = copy(op1)
conj!(tmp)
@test tmp == conj(op1) && conj(tmp.data) == op1.data

# Internal layout
b1 = GenericBasis(2)
b2 = GenericBasis(3)
b3 = GenericBasis(4)
op1 = randoperator(b1, b2)
op2 = randoperator(b2, b3)
x1 = randstate(b2)
d1 = op1.data
d2 = op2.data
v = x1.data
@test (op1*x1).data ≈ [d1[1,1]*v[1] + d1[1,2]*v[2] + d1[1,3]*v[3], d1[2,1]*v[1] + d1[2,2]*v[2] + d1[2,3]*v[3]]
@test (op1*op2).data[2,3] ≈ d1[2,1]*d2[1,3] + d1[2,2]*d2[2,3] + d1[2,3]*d2[3,3]

# Test division
@test 1e-14 > D(op1/7, (1/7)*op1)

# Tensor product
# ==============
op1a = randoperator(b1a, b1b)
op1b = randoperator(b1a, b1b)
op2a = randoperator(b2a, b2b)
op2b = randoperator(b2a, b2b)
op3a = randoperator(b3a, b3b)
op123 = op1a ⊗ op2a ⊗ op3a
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
@test 1e-13 > D((op1a ⊗ op2a) ⊗ op3a, op1a ⊗ (op2a ⊗ op3a))

# Linearity
@test 1e-13 > D(op1a ⊗ (0.3*op2a), 0.3*(op1a ⊗ op2a))
@test 1e-13 > D((0.3*op1a) ⊗ op2a, 0.3*(op1a ⊗ op2a))

# Distributivity
@test 1e-13 > D(op1a ⊗ (op2a + op2b), op1a ⊗ op2a + op1a ⊗ op2b)
@test 1e-13 > D((op2a + op2b) ⊗ op3a, op2a ⊗ op3a + op2b ⊗ op3a)

# Mixed-product property
@test 1e-13 > D((op1a ⊗ op2a) * dagger(op1b ⊗ op2b), (op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)))

# Transpose
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a) ⊗ dagger(op2a))
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a) ⊗ dagger(op2a))

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


# Test identityoperator
x1 = Ket(b_r, rand(ComplexF64, length(b_r)))
x2 = Ket(b_r, rand(ComplexF64, length(b_r)))
xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))
xbra2 = Bra(b_l, rand(ComplexF64, length(b_l)))

I = identityoperator(DenseOpType, b_r)
@test isa(I, DenseOpType)
@test identityoperator(SparseOpType, b_r) == sparse(I)
@test 1e-11 > D(I*x1, x1)
@test I == identityoperator(DenseOpType, b1b) ⊗ identityoperator(DenseOpType, b2b) ⊗ identityoperator(DenseOpType, b3b)

I = identityoperator(DenseOpType, b_l)
@test isa(I, DenseOpType)
@test identityoperator(SparseOpType, b_l) == sparse(I)
@test 1e-11 > D(xbra1*I, xbra1)
@test I == identityoperator(DenseOpType, b1a) ⊗ identityoperator(DenseOpType, b2a) ⊗ identityoperator(DenseOpType, b3a)

# Test tr and normalize
op = DenseOperator(GenericBasis(3), float.([1 3 2;5 2 2;-1 2 5]))
@test 8 == tr(op)
op_normalized = normalize(op)
@test 8 == tr(op)
@test 1 == tr(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test tr(op) != tr(op_copy)
@test 1 ≈ tr(op_copy)
@test op === normalize!(op)

# Test partial tr of state vectors
psi1 = 0.1*randstate(b1a)
psi2 = 0.3*randstate(b2a)
psi3 = 0.7*randstate(b3a)
psi12 = psi1 ⊗ psi2
psi13 = psi1 ⊗ psi3
psi23 = psi2 ⊗ psi3
psi123 = psi1 ⊗ psi2 ⊗ psi3

@test 1e-13 > D(0.1^2*0.3^2*psi3 ⊗ dagger(psi3), ptrace(psi123, [1, 2]))
@test 1e-13 > D(0.1^2*0.7^2*psi2 ⊗ dagger(psi2), ptrace(psi123, [1, 3]))
@test 1e-13 > D(0.3^2*0.7^2*psi1 ⊗ dagger(psi1), ptrace(psi123, [2, 3]))

@test 1e-13 > D(0.1^2*psi23 ⊗ dagger(psi23), ptrace(psi123, 1))
@test 1e-13 > D(0.3^2*psi13 ⊗ dagger(psi13), ptrace(psi123, 2))
@test 1e-13 > D(0.7^2*psi12 ⊗ dagger(psi12), ptrace(psi123, 3))

@test 1e-13 > D(ptrace(psi123, [1, 2]), dagger(ptrace(dagger(psi123), [1, 2])))
@test 1e-13 > D(ptrace(psi123, 3), dagger(ptrace(dagger(psi123), 3)))

@test_throws ArgumentError ptrace(psi123, [1, 2, 3])

@test reduced(psi123, [3]).data == ptrace(psi123, [1, 2]).data

# Test partial tr of operators
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
op1 = randoperator(b1)
op2 = randoperator(b2)
op3 = randoperator(b3)
op123 = op1 ⊗ op2 ⊗ op3

@test 1e-13 > D(op1⊗op2*tr(op3), ptrace(op123, 3))
@test 1e-13 > D(op1⊗op3*tr(op2), ptrace(op123, 2))
@test 1e-13 > D(op2⊗op3*tr(op1), ptrace(op123, 1))

@test 1e-13 > D(op1*tr(op2)*tr(op3), ptrace(op123, [2,3]))
@test 1e-13 > D(op2*tr(op1)*tr(op3), ptrace(op123, [1,3]))
@test 1e-13 > D(op3*tr(op1)*tr(op2), ptrace(op123, [1,2]))

@test_throws ArgumentError ptrace(op123, [1,2,3])
x = randoperator(b1, b1⊗b2)
@test_throws ArgumentError ptrace(x, [1])
x = randoperator(b1⊗b1⊗b2, b1⊗b2)
@test_throws ArgumentError ptrace(x, [1, 2])
x = randoperator(b1⊗b2)
@test_throws ArgumentError ptrace(x, [1, 2])
x = randoperator(b1⊗b2, b2⊗b1)
@test_throws ArgumentError ptrace(x, [1])

op1 = randoperator(b1, b2)
op2 = randoperator(b3)

@test 1e-13 > D(op1*tr(op2), ptrace(op1⊗op2, 2))

# Test expect
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
op1 = randoperator(b1)
op2 = randoperator(b2)
op3 = randoperator(b3)
op123 = op1 ⊗ op2 ⊗ op3
b_l = b1 ⊗ b2 ⊗ b3

state = randstate(b_l)
@test expect(op123, state) ≈ dagger(state)*op123*state
@test expect(1, op1, state) ≈ expect(op1, ptrace(state, [2, 3]))
@test expect(2, op2, state) ≈ expect(op2, ptrace(state, [1, 3]))
@test expect(3, op3, state) ≈ expect(op3, ptrace(state, [1, 2]))

state = randoperator(b_l)
@test expect(op123, state) ≈ tr(op123*state)
@test expect(1, op1, state) ≈ expect(op1, ptrace(state, [2, 3]))
@test expect(2, op2, state) ≈ expect(op2, ptrace(state, [1, 3]))
@test expect(3, op3, state) ≈ expect(op3, ptrace(state, [1, 2]))

@test_throws QuantumOpticsBase.IncompatibleBases expect(op2, op1)

# Permute systems
op1 = randoperator(b1a, b1b)
op2 = randoperator(b2a, b2b)
op3 = randoperator(b3a, b3b)
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


# Test projector
xket = normalize(Ket(b_l, rand(ComplexF64, length(b_l))))
yket = normalize(Ket(b_l, rand(ComplexF64, length(b_l))))
xbra = dagger(xket)
ybra = dagger(yket)

@test 1e-13 > D(projector(xket)*xket, xket)
@test 1e-13 > D(xbra*projector(xket), xbra)
@test 1e-13 > D(projector(xbra)*xket, xket)
@test 1e-13 > D(xbra*projector(xbra), xbra)
@test 1e-13 > D(ybra*projector(yket, xbra), xbra)
@test 1e-13 > D(projector(yket, xbra)*xket, yket)

# Test operator exponential
op = randoperator(b1a)
@test 1e-13 > D(op^2, op*op)
@test 1e-13 > D(op^3, op*op*op)
@test 1e-13 > D(op^4, op*op*op*op)

# Test gemv
b1 = GenericBasis(3)
b2 = GenericBasis(5)
op = randoperator(b1, b2)
xket = randstate(b2)
xbra = dagger(randstate(b1))
rket = randstate(b1)
rbra = dagger(randstate(b2))
alpha = complex(0.7, 1.5)
beta = complex(0.3, 2.1)

rket_ = deepcopy(rket)
QuantumOpticsBase.mul!(rket_,op,xket,complex(1.0),complex(0.))
@test 0 ≈ D(rket_, op*xket)

rket_ = deepcopy(rket)
QuantumOpticsBase.mul!(rket_,op,xket,alpha,beta)
@test 1e-13 > D(rket_, alpha*op*xket + beta*rket)

rbra_ = deepcopy(rbra)
QuantumOpticsBase.mul!(rbra_,xbra,op,complex(1.0),complex(0.))
@test 0 ≈ D(rbra_, xbra*op)

rbra_ = deepcopy(rbra)
QuantumOpticsBase.mul!(rbra_,xbra,op,alpha,beta)
@test 1e-13 > D(rbra_, alpha*xbra*op + beta*rbra)

# # Test gemm
b1 = GenericBasis(37)
b2 = GenericBasis(53)
b3 = GenericBasis(41)
op1 = randoperator(b1, b2)
op2 = randoperator(b2, b3)
r = randoperator(b1, b3)
alpha = complex(0.7, 1.5)
beta = complex(0.3, 2.1)

r_ = deepcopy(r)
QuantumOpticsBase.mul!(r_,op1,op2,complex(1.0),complex(0.))
@test 1e-13 > D(r_, op1*op2)

r_ = deepcopy(r)
QuantumOpticsBase.mul!(r_,op1,op2,alpha,beta)
@test 1e-10 > D(r_, alpha*op1*op2 + beta*r)

dat = rand(prod(b_r.shape))
x = Ket(b_r, dat)
y = Bra(b_r, dat)
@test dm(x) == dm(y)

# Test Hermitian
bspin = SpinBasis(1//2)
bnlevel = NLevelBasis(2)
@test ishermitian(DenseOperator(bspin, bspin, [1.0 im; -im 2.0])) == true
@test ishermitian(DenseOperator(bspin, bnlevel, [1.0 im; -im 2.0])) == false

# Test broadcasting
op1_ = copy(op1)
op1 .= 2*op1
@test op1 == op1_ .+ op1_
op1 .= op1_
@test op1 == op1_
op1 .= op1_ .+ 3 * op1_
@test op1 == 4*op1_
@test_throws DimensionMismatch op1 .= op2
bf = FockBasis(3)
op3 = randoperator(bf)
@test_throws QuantumOpticsBase.IncompatibleBases op1 .+ op3
@test_throws ErrorException cos.(op1)

# Dimension mismatches
b1, b2, b3 = NLevelBasis.((2,3,4))  # N is not a type parameter
@test_throws DimensionMismatch mul!(randstate(b1), randoperator(b2), randstate(b3))
@test_throws DimensionMismatch mul!(randstate(b1)', randstate(b3)', randoperator(b2))
@test_throws DimensionMismatch mul!(randoperator(b1), randoperator(b2), randoperator(b3))
@test_throws DimensionMismatch mul!(randoperator(b1), randoperator(b3)', randoperator(b2))

end # testset

@testset "State-operator tensor products" begin
    b = FockBasis(2) ⊗ SpinBasis(1//2) ⊗ GenericBasis(2)
    b1, b2, b3 = b.bases

    o1 = randoperator(b1)
    v1 = randstate(b1)
    p1 = projector(v1)
    o2 = randoperator(b2)
    v2 = randstate(b2)
    p2 = projector(v2)
    o3 = randoperator(b3)
    v3 = randstate(b3)
    p3 = projector(v3)

    @test (v1 ⊗ o2).basis_l == b1 ⊗ b2
    @test (v1 ⊗ o2).basis_r == b2
    @test (v1' ⊗ o2).basis_l == b2
    @test (v1' ⊗ o2).basis_r == b1 ⊗ b2

    @test ((o1 ⊗ v2) * (o1 ⊗ v2')).data ≈ (o1^2 ⊗ p2).data
    @test ((o1 ⊗ v2') * (o1 ⊗ v2)).data ≈ (o1^2).data

    @test ((o1 ⊗ v2 ⊗ o3) * (o1 ⊗ v2' ⊗ o3)).data ≈ (o1^2 ⊗ p2 ⊗ o3^2).data
    @test ((v1 ⊗ o2 ⊗ o3) * (v1' ⊗ o2 ⊗ o3)).data ≈ (p1 ⊗ o2^2 ⊗ o3^2).data
    @test ((v1 ⊗ o2 ⊗ v3) * (v1' ⊗ o2 ⊗ v3')).data ≈ (p1 ⊗ o2^2 ⊗ p3).data
end
