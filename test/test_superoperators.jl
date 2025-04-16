using Test
using QuantumOpticsBase
using SparseArrays, LinearAlgebra

@testset "superoperators" begin

# Test length
b1 = GenericBasis(3)
op = DenseOperator(b1, b1)
S = spre(op)
@test length(S) == length(S.data) == (3*3)^2
S = spost(op)
@test length(S) == length(S.data) == (3*3)^2
b2 = GenericBasis(5)
op = DenseOperator(b1, b2)
S = sprepost(dagger(op), op)
@test length(S) == length(S.data) == (3*5)^2

# Test arithmetic
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(5)
b4 = GenericBasis(3)

d1 = DenseOperator(KetBraBasis(b1, b2), KetBraBasis(b3, b4))
d2 = DenseOperator(KetBraBasis(b3, b4), KetBraBasis(b1, b2))
s1 = SparseOperator(KetBraBasis(b1, b2), KetBraBasis(b3, b4))
s2 = SparseOperator(KetBraBasis(b3, b4), KetBraBasis(b1, b2))

x = d1*d2
@test isa(x, DenseOpType)
@test x.basis_l == x.basis_r == KetBraBasis(b1, b2)

x = s1*s2
@test isa(x, SparseOpType)
@test x.basis_l == x.basis_r == KetBraBasis(b1, b2)

x = d1*s2
@test isa(x, DenseOpType)
@test x.basis_l == x.basis_r == KetBraBasis(b1, b2)

x = s1*d2
@test isa(x, DenseOpType)
@test x.basis_l == x.basis_r == KetBraBasis(b1, b2)

x = d1*3
@test isa(x, DenseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = 2.5*s1
@test isa(x, SparseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = d1 + d1
@test isa(x, DenseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = s1 + s1
@test isa(x, SparseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = d1 + s1
@test isa(x, DenseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = d1 - d1
@test isa(x, DenseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = s1 - s1
@test isa(x, SparseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = d1 - s1
@test isa(x, DenseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = -d1
@test isa(x, DenseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

x = -s1
@test isa(x, SparseOpType)
@test x.basis_l == KetBraBasis(b1, b2)
@test x.basis_r == KetBraBasis(b3, b4)

# TODO: Clean-up this part
ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

T = Float64[0.,1.]

fockbasis = FockBasis(7)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
H = Ha + Hc + Hint

Ja = embed(basis, 1, sqrt(γ)*sm)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
ρ₀ = dm(Ψ₀)

@test identitysuperoperator(spinbasis)*sx == sx
@test choi(identitysuperoperator(spinbasis))*sx == sx
@test identitysuperoperator(sparse(spre(sx)))*sx == sparse(sx)
@test choi(identitysuperoperator(sparse(spre(sx))))*sx == sparse(sx)
@test sparse(choi(identitysuperoperator(spre(sx))))*sx == sparse(sx)
@test identitysuperoperator(dense(spre(sx)))*sx == dense(sx)
@test choi(identitysuperoperator(dense(spre(sx))))*sx == dense(sx)
@test dense(choi(identitysuperoperator(spre(sx))))*sx == dense(sx)

op1 = DenseOperator(spinbasis, [1.2+0.3im 0.7+1.2im;0.3+0.1im 0.8+3.2im])
op2 = DenseOperator(spinbasis, [0.2+0.1im 0.1+2.3im; 0.8+4.0im 0.3+1.4im])
@test tracedistance(spre(op1)*op2, op1*op2) < 1e-12
@test tracedistance(choi(spre(op1))*op2, op1*op2) < 1e-12
@test tracedistance(spost(op1)*op2, op2*op1) < 1e-12
@test tracedistance(choi(spost(op1))*op2, op2*op1) < 1e-12

@test spre(sparse(op1))*op2 ≈ op1*op2
@test choi(spre(sparse(op1)))*op2 ≈ op1*op2
@test spost(sparse(op1))*op2 ≈ op2*op1
@test choi(spost(sparse(op1)))*op2 ≈ op2*op1
@test spre(sparse(dagger(op1)))*op2 ≈ dagger(op1)*op2
@test choi(spre(sparse(dagger(op1))))*op2 ≈ dagger(op1)*op2
@test spre(dense(dagger(op1)))*op2 ≈ dagger(op1)*op2
@test choi(spre(dense(dagger(op1))))*op2 ≈ dagger(op1)*op2
@test sprepost(sparse(op1), op1)*op2 ≈ op1*op2*op1
@test choi(sprepost(sparse(op1), op1))*op2 ≈ op1*op2*op1

@test spre(sparse(op1))*sparse(op2) == sparse(op1*op2)
@test choi(spre(sparse(op1)))*sparse(op2) == sparse(op1*op2)
@test spost(sparse(op1))*sparse(op2) == sparse(op2*op1)
@test choi(spost(sparse(op1)))*sparse(op2) == sparse(op2*op1)
@test sprepost(sparse(op1), sparse(op1))*sparse(op2) ≈ sparse(op1*op2*op1)
@test choi(sprepost(sparse(op1), sparse(op1)))*sparse(op2) ≈ sparse(op1*op2*op1)

@test sprepost(op1, op2) ≈ spre(op1)*spost(op2)
b1 = FockBasis(1)
b2 = FockBasis(5)
op = fockstate(b1, 0) ⊗ dagger(fockstate(b2, 0))
@test sprepost(dagger(op), op)*dm(fockstate(b1, 0)) == dm(fockstate(b2, 0))
@test choi(sprepost(dagger(op), op))*dm(fockstate(b1, 0)) == dm(fockstate(b2, 0))
@test_throws ArgumentError spre(op)
@test_throws ArgumentError spost(op)

L = liouvillian(H, J)
ρ = -1im*(H*ρ₀ - ρ₀*H)
for j=J
    ρ .+= j*ρ₀*dagger(j) - 0.5*dagger(j)*j*ρ₀ - 0.5*ρ₀*dagger(j)*j
end
@test tracedistance(L*ρ₀, ρ) < 1e-10
@test tracedistance(choi(L)*ρ₀, ρ) < 1e-10

# tout, ρt = timeevolution.master([0.,1.], ρ₀, H, J; reltol=1e-7)
# @test tracedistance(exp(dense(L))*ρ₀, ρt[end]) < 1e-6
# @test tracedistance(exp(sparse(L))*ρ₀, ρt[end]) < 1e-6

@test dense(spre(op1)) == spre(op1)
@test dense(choi(spre(op1))) == choi(spre(op1))

@test L/2.0 == 0.5*L == L*0.5
@test -L == SparseOperator(L.basis_l, L.basis_r, -L.data)

@test_throws AssertionError liouvillian(H, J; rates=zeros(4, 4))

rates = diagm(0 => [1.0, 1.0])
@test liouvillian(H, J; rates=rates) == L

# Test broadcasting
@test L .+ L == L + L
Ldense = dense(L)
# @test isa(L .+ Ldense, DenseSuperOpType) # Broadcasting of sparse .+ dense returns sparse
L_ = copy(L)
L .+= L
@test L == 2*L_
L .+= Ldense
@test L == 3*L_
Ldense_ = dense(L_)
Ldense .+= Ldense
@test Ldense == 2*Ldense_
Ldense .+= L
@test isa(Ldense, DenseOpType)
@test isapprox(Ldense.data, 5*Ldense_.data)

b = FockBasis(20)
L = liouvillian(identityoperator(b), [destroy(b)])
@test exp(sparse(L)).data ≈ exp(dense(L)).data
@test exp(sparse(zero(identitysuperoperator(b)))).data ≈ identitysuperoperator(b).data
N = exp(log(2) * sparse(L)) # 50% loss channel
ρ = N * dm(coherentstate(b, 1))
@test (1 - real(tr(ρ^2))) < 1e-10 # coherent state remains pure under loss
@test tracedistance(ρ, dm(coherentstate(b, 1/sqrt(2)))) < 1e-10
ρ = N * dm(fockstate(b, 1))
@test (0.5 - real(tr(ρ^2))) < 1e-10 # one photon state becomes maximally mixed
@test tracedistance(ρ, normalize(dm(fockstate(b, 0)) + dm(fockstate(b, 1)))) < 1e-10

# Testing 0-2-4 binomial code encoder
b_logical = SpinBasis(1//2)
b_fock = FockBasis(5)
z_l = normalize(fockstate(b_fock, 0) + 1im*fockstate(b_fock, 4))
o_l = fockstate(b_fock, 2)
encoder_kraus = z_l ⊗ dagger(spinup(b_logical)) + o_l ⊗ dagger(spindown(b_logical))
encoder_sup = sprepost(encoder_kraus, dagger(encoder_kraus))
decoder_sup = sprepost(dagger(encoder_kraus), encoder_kraus)
@test super(choi(encoder_sup)).data == encoder_sup.data
@test decoder_sup == dagger(encoder_sup)
@test choi(decoder_sup) == dagger(choi(encoder_sup))
@test decoder_sup*encoder_sup ≈ dense(identitysuperoperator(b_logical))
@test decoder_sup*choi(encoder_sup) ≈ dense(identitysuperoperator(b_logical))
@test choi(decoder_sup)*encoder_sup ≈ dense(identitysuperoperator(b_logical))
@test super(choi(decoder_sup)*choi(encoder_sup)) ≈ dense(identitysuperoperator(b_logical))

end # testset
