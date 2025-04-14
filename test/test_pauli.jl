using LinearAlgebra
using Test

using QuantumOpticsBase

@testset "pauli" begin

bs = SpinBasis(1//2)
bp = PauliBasis(1)

# Test Pauli basis vectors are in I, X, Y, Z order
@test pauli(identityoperator(bs)/sqrt(2)).data ≈ [1., 0, 0, 0]
@test pauli(sigmax(bs)/sqrt(2)).data ≈ [0, 1., 0, 0]
@test pauli(sigmay(bs)/sqrt(2)).data ≈ [0, 0, 1., 0]
@test pauli(sigmaz(bs)/sqrt(2)).data ≈ [0, 0, 0, 1.]

# Test that single qubit unitary Pauli channels are diagonal
Isop = sprepost(identityoperator(bs), dagger(identityoperator(bs)))
Xsop = sprepost(sigmax(bs), dagger(sigmax(bs)))
Ysop = sprepost(sigmay(bs), dagger(sigmay(bs)))
Zsop = sprepost(sigmaz(bs), dagger(sigmaz(bs)))
@test pauli(Isop).data ≈ diagm([1, 1, 1, 1])
@test pauli(Xsop).data ≈ diagm([1, 1, -1, -1])
@test pauli(Ysop).data ≈ diagm([1, -1, 1, -1])
@test pauli(Zsop).data ≈ diagm([1, -1, -1, 1])

# Test bit flip encoder isometry
encoder_kraus = (tensor_pow(spinup(bs), 3) ⊗ dagger(spinup(bs)) +
                 tensor_pow(spindown(bs), 3) ⊗ dagger(spindown(bs)))
encoder_sup = sprepost(encoder_kraus, dagger(encoder_kraus))
decoder_sup = sprepost(dagger(encoder_kraus), encoder_kraus)
@test super(choi(encoder_sup)).data == encoder_sup.data
@test decoder_sup == dagger(encoder_sup)
@test choi(decoder_sup) == dagger(choi(encoder_sup))
@test decoder_sup*encoder_sup ≈ dense(identitysuperoperator(bs))
@test decoder_sup*choi(encoder_sup) ≈ dense(identitysuperoperator(bs))
@test choi(decoder_sup)*encoder_sup ≈ dense(identitysuperoperator(bs))
@test super(choi(decoder_sup)*choi(encoder_sup)) ≈ dense(identitysuperoperator(bs))

# Test conversion of unitary matrices to superoperators.
CZ = dm(spinup(bs))⊗identityoperator(bs) + dm(spindown(bs))⊗sigmaz(bs)
CZ_sop = sprepost(CZ,dagger(CZ))

@test CZ_sop.data ≈ diagm([1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1])
@test basis_l(CZ_sop) == basis_r(CZ_sop) == KetBraBasis(bs^2, bs^2)

# Test conversion of superoperator to Pauli transfer matrix.
CZ_ptm = pauli(CZ_sop)

# Test construction of dense Pauli transfer matrix
@test_throws DimensionMismatch Operator(PauliBasis(2), PauliBasis(3), CZ_ptm.data)
@test Operator(PauliBasis(2), PauliBasis(2), CZ_ptm.data) == CZ_ptm

@test all(isapprox.(CZ_ptm.data[[1,30,47,52,72,91,117,140,166,185,205,210,227,256]], 1))
@test all(isapprox.(CZ_ptm.data[[106,151]], -1))

@test CZ_ptm == pauli(chi(CZ_sop))

CNOT = dm(spinup(bs))⊗identityoperator(bs) + dm(spindown(bs))⊗sigmax(bs)
CNOT_sop = sprepost(CZ,dagger(CZ))
CNOT_chi = chi(CNOT_sop)
CNOT_ptm = pauli(CNOT_sop)

@test basis_l(CNOT_sop) == basis_r(CNOT_sop) == KetBraBasis(bs^2, bs^2)
@test basis_l(CNOT_chi) == basis_r(CNOT_chi) == ChoiBasis(bp^2, bp^2)
@test basis_l(CNOT_ptm) == basis_r(CNOT_ptm) == ChoiBasis(bp^2, bp^2)

@test all(isapprox.(imag.(CNOT_sop.data), 0))
@test all(isapprox.(imag.(CNOT_chi.data), 0))
@test all(isapprox.(imag.(CNOT_ptm.data), 0))

@test all(isapprox.(CNOT_sop.data[[1,18,36,51,69,86,104,119,141,158,176,191,201,218,236,251]], 1))
@test all(isapprox.(CNOT_chi.data[[1,2,13,17,18,29,193,194,205,222]], 1))
@test all(isapprox.(CNOT_chi.data[[14,30,206,209,210,221]], -1))
@test all(isapprox.(CNOT_ptm.data[[1,18,47,64,70,85,108,138,153,183,205,222,227,244,]], 1))
@test all(isapprox.(CNOT_ptm.data[[123,168]], -1))

# Test construction of chi matrix
@test_throws DimensionMismatch Operator(ChoiBasis(bp^2, bp^2), ChoiBasis(bp^3, bp^3), CNOT_chi.data)
@test Operator(ChoiBasis(bp^2, bp^2), CNOT_chi.data) == CNOT_chi

# Test equality and conversion of identity among all three bases.
ident = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

IDENT = DenseOperator(bs^2, ident)

IDENT_sop = SuperOperator(IDENT)
IDENT_chi = ChiMatrix(IDENT)
IDENT_ptm = PauliTransferMatrix(IDENT)

@test chi(IDENT_sop) == IDENT_chi
@test chi(IDENT_ptm) == IDENT_chi
@test super(IDENT_chi) == IDENT_sop
@test super(IDENT_ptm) == IDENT_sop
@test pauli(IDENT_sop) == IDENT_ptm
@test pauli(IDENT_chi) == IDENT_ptm

# Test approximate equality and conversion among all three bases.
cphase = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(1im*.6)]

CPHASE = DenseOperator(b^2, cphase)

CPHASE_sop = SuperOperator(CPHASE)
CPHASE_chi = ChiMatrix(CPHASE)
CPHASE_ptm = PauliTransferMatrix(CPHASE)

@test isapprox(ChiMatrix(CPHASE_sop), CPHASE_chi)
@test isapprox(ChiMatrix(CPHASE_ptm), CPHASE_chi)
@test isapprox(SuperOperator(CPHASE_chi), CPHASE_sop)
@test isapprox(SuperOperator(CPHASE_ptm), CPHASE_sop)
@test isapprox(PauliTransferMatrix(CPHASE_sop), CPHASE_ptm)
@test isapprox(PauliTransferMatrix(CPHASE_chi), CPHASE_ptm)

# Test composition.
@test isapprox(ChiMatrix(CPHASE) * ChiMatrix(CNOT), ChiMatrix(CPHASE * CNOT))
@test isapprox(PauliTransferMatrix(CPHASE) * PauliTransferMatrix(CNOT), PauliTransferMatrix(CPHASE * CNOT))

end # testset
