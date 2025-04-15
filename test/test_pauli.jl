using LinearAlgebra
using Test

using QuantumOpticsBase
using QuantumOpticsBase: pauli_comp

@testset "pauli" begin

bs = SpinBasis(1//2)
bp = PauliBasis(1)

for N=1:5
    @test (dagger(pauli_comp(N))*pauli_comp(N)) ≈ identityoperator(PauliBasis(N))
    @test (pauli_comp(N)*dagger(pauli_comp(N))) ≈ identityoperator(KetBraBasis(bs^N, bs^N))
end

I = identityoperator(bs)
X = sigmax(bs)
Y = sigmay(bs)
Z = sigmaz(bs)
Isop = sprepost(I, dagger(I))
Xsop = sprepost(X, dagger(X))
Ysop = sprepost(Y, dagger(Y))
Zsop = sprepost(Z, dagger(Z))

# Test Pauli basis vectors are in I, X, Y, Z order
@test pauli(I/sqrt(2)).data ≈ [1., 0, 0, 0]
@test pauli(X/sqrt(2)).data ≈ [0, 1., 0, 0]
@test pauli(Y/sqrt(2)).data ≈ [0, 0, 1., 0]
@test pauli(Z/sqrt(2)).data ≈ [0, 0, 0, 1.]

# Test that single qubit unitary Pauli channels are diagonal
@test pauli(Isop).data ≈ diagm([1, 1, 1, 1])
@test pauli(Xsop).data ≈ diagm([1, 1, -1, -1])
@test pauli(Ysop).data ≈ diagm([1, -1, 1, -1])
@test pauli(Zsop).data ≈ diagm([1, -1, -1, 1])
@test chi(Isop).data ≈ diagm([1, 0, 0, 0])
@test chi(Xsop).data ≈ diagm([0, 1, 0, 0])
@test chi(Ysop).data ≈ diagm([0, 0, 1, 0])
@test chi(Zsop).data ≈ diagm([0, 0, 0, 1])

# Test Haddamard Clifford rules
H = ( (spinup(bs)+spindown(bs))⊗dagger(spinup(bs)) +
    (spinup(bs)-spindown(bs))⊗dagger(spindown(bs)) )/sqrt(2)
Hsop = sprepost(H, dagger(H))
@test Hsop*I ≈ I
@test Hsop*X ≈ Z
@test Hsop*Y ≈ -Y
@test Hsop*Z ≈ X
@test pauli(Hsop)*I ≈ I
@test pauli(Hsop)*X ≈ Z
@test pauli(Hsop)*Y ≈ -Y
@test pauli(Hsop)*Z ≈ X
@test chi(Hsop)*I ≈ I
@test chi(Hsop)*X ≈ Z
@test chi(Hsop)*Y ≈ -Y
@test chi(Hsop)*Z ≈ X
@test pauli(Hsop).data ≈ diagm(0=>[1,0,-1,0], 2=>[0,1], -2=>[0,1]) 
@test chi(Hsop).data ≈ diagm(0=>[0,1,0,1], 2=>[0,1], -2=>[0,1]) 

    """
# Test bit flip encoder isometry
encoder_kraus = (tensor_pow(spinup(bs), 3) ⊗ dagger(spinup(bs)) +
                 tensor_pow(spindown(bs), 3) ⊗ dagger(spindown(bs)))
encoder_sup = sprepost(encoder_kraus, dagger(encoder_kraus))
decoder_sup = sprepost(dagger(encoder_kraus), encoder_kraus)
for f1 in [super, choi, pauli, chi]
    @test f1(decoder_sup) ≈ f1(dagger(encoder_sup))
    @test f1(decoder_sup) ≈ dagger(f1(encoder_sup))
    for f2 in [super, choi, pauli, chi]
        @test super(f2(f1(encoder_sup))).data ≈ encoder_sup.data
        @test f2(decoder_sup)*f1(encoder_sup) ≈ dense(identitysuperoperator(bs))
    end
end

# Test conversion of unitary matrices to superoperators.
CZ = dm(spinup(bs))⊗identityoperator(bs) + dm(spindown(bs))⊗sigmaz(bs)
CZ_sop = sprepost(CZ,dagger(CZ))

@test basis_l(CZ_sop) == basis_r(CZ_sop) == KetBraBasis(bs^2, bs^2)
@test CZ_sop*(I⊗I) ≈ I⊗I
@test CZ_sop*(Z⊗I) ≈ X⊗I
@test CZ_sop*(I⊗Z) ≈ I⊗Z
@test CZ_sop*(I⊗X) ≈ X⊗X

# Test conversion of superoperator to Pauli transfer matrix.
CZ_ptm = pauli(CZ_sop)

# Test construction of dense Pauli transfer matrix
@test_throws DimensionMismatch Operator(PauliBasis(2), PauliBasis(3), CZ_ptm.data)
@test Operator(PauliBasis(2), PauliBasis(2), CZ_ptm.data) == CZ_ptm
@test CZ_ptm*pauli(I⊗I) ≈ pauli(I⊗I)
@test CZ_ptm*pauli(Z⊗I) ≈ pauli(X⊗I)
@test CZ_ptm*pauli(I⊗Z) ≈ pauli(I⊗Z)
@test CZ_ptm*pauli(I⊗X) ≈ pauli(X⊗X)

@test CZ_ptm ≈ pauli(chi(CZ_sop))
"""

CNOT = dm(spinup(bs))⊗identityoperator(bs) + dm(spindown(bs))⊗sigmax(bs)
CNOT_sop = sprepost(CNOT,dagger(CNOT))
CNOT_choi = choi(CNOT_sop)
CNOT_chi = chi(CNOT_sop)
CNOT_ptm = pauli(CNOT_sop)

@test basis_l(CNOT_sop) == basis_r(CNOT_sop) == KetBraBasis(bs^2, bs^2)
@test basis_l(CNOT_chi) == basis_r(CNOT_chi) == ChiBasis(2)
@test basis_l(CNOT_ptm) == basis_r(CNOT_ptm) == PauliBasis(2)

@test all(isapprox.(imag.(CNOT_sop.data), 0))
@test all(isapprox.(imag.(CNOT_chi.data), 0))
@test all(isapprox.(imag.(CNOT_ptm.data), 0))

for op in (CNOT_sop, CNOT_choi, CNOT_chi, CNOT_ptm)
    @test op*(I⊗I) ≈ I⊗I
    @test op*(X⊗I) ≈ X⊗X
    @test op*(Z⊗I) ≈ Z⊗I
    @test op*(I⊗X) ≈ I⊗X
    @test op*(I⊗Z) ≈ Z⊗Z
end

# Test construction of chi matrix
@test_throws DimensionMismatch Operator(ChiBasis(bs^2, bs^2), ChiBasis(bs^3, bs^3), CNOT_chi.data)
@test Operator(ChiBasis(bs^2, bs^2), CNOT_chi.data) == CNOT_chi

# Test equality and conversion of identity among all three bases.
IDENT_sop = identitysuperoperator(bs^2)
IDENT_chi = chi(IDENT_sop)
IDENT_ptm = pauli(IDENT_sop)

@test IDENT_sop.data ≈ IDENT_ptm.data
@test chi(IDENT_ptm).data ≈ choi(IDENT_sop).data
@test chi(IDENT_ptm) ≈ IDENT_chi
@test super(IDENT_chi) ≈ IDENT_sop
@test super(IDENT_ptm) ≈ IDENT_sop
@test pauli(IDENT_sop) ≈ IDENT_ptm
@test pauli(IDENT_chi) ≈ IDENT_ptm

# Test approximate equality and conversion among all three bases.
cphase(θ) = dm(spinup(bs))⊗identityoperator(bs) +
    dm(spindown(bs))⊗(spindown(bs)⊗spindown(bs)' + exp(1im*θ)spindown(bs)⊗spindown(bs)')

CPHASE_sop = sprepost(cphase(0.6),dagger(cphase(0.6)))
CPHASE_chi = chi(CPHASE_sop)
CPHASE_ptm = pauli(CPHASE_sop)

@test isapprox(chi(CPHASE_sop), CPHASE_chi)
@test isapprox(chi(CPHASE_ptm), CPHASE_chi)
@test isapprox(super(CPHASE_chi), CPHASE_sop)
@test isapprox(super(CPHASE_ptm), CPHASE_sop)
@test isapprox(pauli(CPHASE_sop), CPHASE_ptm)
@test isapprox(pauli(CPHASE_chi), CPHASE_ptm)

# Test composition.
@test isapprox(chi(CPHASE_sop) * chi(CNOT_sop), chi(CPHASE_sop * CNOT_sop))
@test isapprox(pauli(CPHASE_sop) * pauli(CNOT_sop), pauli(CPHASE_sop * CNOT_sop))

end # testset
