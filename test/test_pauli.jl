using LinearAlgebra
using Test

using QuantumOpticsBase
import QuantumOpticsBase: comp_pauli_kb

@testset "pauli" begin

b = SpinBasis(1//2)
Isop = sprepost(identityoperator(b), dagger(identityoperator(b)))
Xsop = sprepost(sigmax(b), dagger(sigmax(b)))
Ysop = sprepost(sigmay(b), dagger(sigmay(b)))
Zsop = sprepost(sigmaz(b), dagger(sigmaz(b)))
Xsk = vec(sigmax(b))
V = comp_pauli_kb(1)


# Test conversion of unitary matrices to superoperators.
CZ = dm(spinup(b))⊗identityoperator(b) + dm(spindown(b))⊗sigmaz(b)
CZ_sop = sprepost(CZ,dagger(CZ))

# Test conversion of unitary matrices to superoperators.
@test diag(CZ_sop.data) ==  ComplexF64[1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1]
@test basis_l(CZ_sop) == basis_r(CZ_sop) == KetBraBasis(b^2, b^2)

# Test conversion of superoperator to Pauli transfer matrix.
CZ_ptm = pauli(CZ_sop)

# Test DensePauliTransferMatrix constructor.
@test_throws DimensionMismatch Operator(PauliBasis(2), PauliBasis(3), CZ_ptm.data)
@test Operator(PauliBasis(2), PauliBasis(2), CZ_ptm.data) == CZ_ptm

@test all(isapprox.(CZ_ptm.data[[1,30,47,52,72,91,117,140,166,185,205,210,227,256]], 4))
@test all(isapprox.(CZ_ptm.data[[106,151]], -4))

@test CZ_ptm == PauliTransferMatrix(ChiMatrix(CZ))

# Test construction of non-symmetric unitary.
CNOT = DenseOperator(b^2, b^2, diagm(0 => [1,1,0,0], 1 => [0,0,1], -1 => [0,0,1]))
CNOT_sop = SuperOperator(CNOT)
CNOT_chi = ChiMatrix(CNOT)
CNOT_ptm = PauliTransferMatrix(CNOT)

@test CNOT_sop.basis_l == CNOT_sop.basis_r == (b^2, b^2)
@test CNOT_chi.basis_l == CNOT_chi.basis_r == (b^2, b^2)
@test CNOT_ptm.basis_l == CNOT_ptm.basis_r == (b^2, b^2)

@test all(isapprox.(imag.(CNOT_sop.data), 0))
@test all(isapprox.(imag.(CNOT_chi.data), 0))
@test all(isapprox.(imag.(CNOT_ptm.data), 0))

@test all(isapprox.(CNOT_sop.data[[1,18,36,51,69,86,104,119,141,158,176,191,201,218,236,251]], 1))
@test all(isapprox.(CNOT_chi.data[[1,2,13,17,18,29,193,194,205,222]], 1))
@test all(isapprox.(CNOT_chi.data[[14,30,206,209,210,221]], -1))
@test all(isapprox.(CNOT_ptm.data[[1,18,47,64,70,85,108,138,153,183,205,222,227,244,]], 1))
@test all(isapprox.(CNOT_ptm.data[[123,168]], -1))

# Test DenseChiMatrix constructor.
@test_throws DimensionMismatch Operator(ChoiBasis(PauliBasis(2), PauliBasis(2)), ChoiBasis(PauliBasis(3), PauliBasis(3)), CNOT_chi.data)
@test Operator(ChoiBasis(PauliBasis(2), PauliBasis(2)), CNOT_chi.data) == CNOT_chi

# Test equality and conversion among all three bases.
ident = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

IDENT = DenseOperator(b^2, ident)

IDENT_sop = SuperOperator(IDENT)
IDENT_chi = ChiMatrix(IDENT)
IDENT_ptm = PauliTransferMatrix(IDENT)

@test ChiMatrix(IDENT_sop) == IDENT_chi
@test ChiMatrix(IDENT_ptm) == IDENT_chi
@test SuperOperator(IDENT_chi) == IDENT_sop
@test SuperOperator(IDENT_ptm) == IDENT_sop
@test PauliTransferMatrix(IDENT_sop) == IDENT_ptm
@test PauliTransferMatrix(IDENT_chi) == IDENT_ptm

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
