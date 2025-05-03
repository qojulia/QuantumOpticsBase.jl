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

# Test that single qubit unitary channels are diagonal in Pauli transfer and chi
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
H_sop = sprepost(H, dagger(H))
@test pauli(H_sop).data ≈ diagm(0=>[1,0,-1,0], 2=>[0,1], -2=>[0,1]) 
@test chi(H_sop).data ≈ diagm(0=>[0,1,0,1], 2=>[0,1], -2=>[0,1])/2

for op in (H_sop, choi(H_sop), pauli(H_sop), chi(H_sop))
    @test op*I ≈ I
    @test op*X ≈ Z
    @test op*Y ≈ -Y
    @test op*Z ≈ X
end

# Test equality and conversion of identity among all three bases.
IDENT_sop = identitysuperoperator(bs^2)
IDENT_chi = chi(IDENT_sop)
IDENT_ptm = pauli(IDENT_sop)

@test IDENT_sop.data ≈ IDENT_ptm.data
@test chi(IDENT_ptm) ≈ IDENT_chi
@test super(IDENT_chi) ≈ IDENT_sop
@test super(IDENT_ptm) ≈ IDENT_sop
@test pauli(IDENT_sop) ≈ IDENT_ptm
@test pauli(IDENT_chi) ≈ IDENT_ptm

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
        @test super(f2(decoder_sup)*f1(encoder_sup)) ≈ dense(identitysuperoperator(bs))
    end
end

# Test CZ and CNOT
CZ = dm(spinup(bs))⊗identityoperator(bs) + dm(spindown(bs))⊗sigmaz(bs)
CNOT = dm(spinup(bs))⊗identityoperator(bs) + dm(spindown(bs))⊗sigmax(bs)
CZ_rules =   [(I⊗I, I⊗I), (X⊗I, X⊗Z), (Z⊗I, Z⊗I), (I⊗X, I⊗Z), (I⊗Z, X⊗Z)]
CNOT_rules = [(I⊗I, I⊗I), (X⊗I, X⊗X), (Z⊗I, Z⊗I), (I⊗X, I⊗X), (I⊗Z, Z⊗Z)]

#for (gate, rules) in [(CZ, CZ_rules), (CNOT, CNOT_rules)]
for (gate, rules) in [(CNOT, CNOT_rules)]
    op_sop = sprepost(gate,dagger(gate))
    op_choi = choi(op_sop)
    op_ptm = pauli(op_sop)
    op_chi = chi(op_sop)

    @test_throws DimensionMismatch Operator(PauliBasis(2), PauliBasis(3), op_ptm.data)
    @test_throws DimensionMismatch Operator(ChiBasis(2, 2), ChiBasis(3, 3), op_chi.data)
    @test Operator(PauliBasis(2), PauliBasis(2), op_ptm.data) == op_ptm
    @test Operator(ChiBasis(2, 2), op_chi.data) == op_chi
    @test basis_l(op_sop) == basis_r(op_sop) == KetBraBasis(bs^2, bs^2)
    @test basis_l(op_choi) == basis_r(op_choi) == ChoiBasis(bs^2, bs^2)
    @test basis_l(op_chi) == basis_r(op_chi) == ChiBasis(2, 2)
    @test basis_l(op_ptm) == basis_r(op_ptm) == PauliBasis(2)

    @test all(isapprox.(imag.(op_sop.data), 0))
    @test all(isapprox.(imag.(op_choi.data), 0))
    @test all(isapprox.(imag.(op_ptm.data), 0; atol=1e-26))
    @test dagger(op_chi) ≈ op_chi

    for (lhs, rhs) in rules
        @test op_ptm*pauli(lhs) ≈ pauli(rhs)
        for op in (op_sop, op_choi, op_chi, op_ptm)
            @test op*lhs ≈ rhs
        end
    end
end

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
CNOT_sop = sprepost(CNOT,dagger(CNOT))
@test isapprox(chi(CPHASE_sop) * chi(CNOT_sop), chi(CPHASE_sop * CNOT_sop))
@test isapprox(pauli(CPHASE_sop) * pauli(CNOT_sop), pauli(CPHASE_sop * CNOT_sop))

end # testset
