@testitem "test_pauli" begin
using LinearAlgebra
using Test

using QuantumOpticsBase

@testset "pauli" begin

b = SpinBasis(1//2)
# Test conversion of unitary matrices to superoperators.
q2 = b^2
q3 = b^3

@testset "Pauli multiplication helper" begin
    paulis = (
        ComplexF64[1 0; 0 1],
        ComplexF64[0 1; 1 0],
        ComplexF64[0 -im; im 0],
        ComplexF64[1 0; 0 -1],
    )
    for left in 0:3, right in 0:3
        product, phase = @inferred QuantumOpticsBase.multiply_pauli_matrices(
            string(left), string(right),
        )
        product_digit = xor(left, right)
        @test product == string(product_digit)
        @test phase * paulis[product_digit+1] == paulis[left+1] * paulis[right+1]
    end

    @test QuantumOpticsBase.multiply_pauli_matrices("010", "023") == ("033", 1im)
    @test QuantumOpticsBase.multiply_pauli_matirices("010", "023") == ("033", 1im)
    @test_throws ArgumentError QuantumOpticsBase.multiply_pauli_matrices("", "")
    @test_throws ArgumentError QuantumOpticsBase.multiply_pauli_matrices("0", "00")
    @test_throws ArgumentError QuantumOpticsBase.multiply_pauli_matrices("4", "0")
    @test_throws ArgumentError QuantumOpticsBase.multiply_pauli_matrices("0", "x")
end

@testset "Concurrent Chi-matrix composition" begin
    left_data = reshape(ComplexF64.(1:16), 4, 4)
    right_data = reshape(ComplexF64.(16:-1:1), 4, 4)
    left = DenseChiMatrix((b, b), (b, b), left_data)
    right = DenseChiMatrix((b, b), (b, b), right_data)
    left_before = copy(left.data)
    right_before = copy(right.data)
    num_compositions = max(8, 4 * min(Threads.nthreads(), 8))
    tasks = [Threads.@spawn(left * right) for _ in 1:num_compositions]
    results = fetch.(tasks)

    @test all(result -> result == first(results), results)
    @test left.data == left_before
    @test right.data == right_before
end

CZ = DenseOperator(q2, q2, diagm(0 => [1,1,1,-1]))
CZ_sop = SuperOperator(CZ)

# Test conversion of unitary matrices to superoperators.
@test diag(CZ_sop.data) ==  ComplexF64[1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1]
@test CZ_sop.basis_l == CZ_sop.basis_r == (q2, q2)

# Test conversion of superoperator to Pauli transfer matrix.
CZ_ptm = PauliTransferMatrix(CZ_sop)

# Test DensePauliTransferMatrix constructor.
@test_throws DimensionMismatch DensePauliTransferMatrix((q2, q2), (q3, q3), CZ_ptm.data)
@test DensePauliTransferMatrix((q2, q2), (q2, q2), CZ_ptm.data) == CZ_ptm

@test all(isapprox.(CZ_ptm.data[[1,30,47,52,72,91,117,140,166,185,205,210,227,256]], 1))
@test all(isapprox.(CZ_ptm.data[[106,151]], -1))

@test CZ_ptm == PauliTransferMatrix(ChiMatrix(CZ))

# Test construction of non-symmetric unitary.
CNOT = DenseOperator(q2, q2, diagm(0 => [1,1,0,0], 1 => [0,0,1], -1 => [0,0,1]))
CNOT_sop = SuperOperator(CNOT)
CNOT_chi = @inferred ChiMatrix(CNOT)
CNOT_ptm = @inferred PauliTransferMatrix(CNOT)

pauli_vectors = @inferred QuantumOpticsBase.pauli_basis_vectors(2)
@test pauli_vectors' * pauli_vectors ≈ 4I
@test_throws ArgumentError QuantumOpticsBase.pauli_basis_vectors(0)

@test CNOT_sop.basis_l == CNOT_sop.basis_r == (q2, q2)
@test CNOT_chi.basis_l == CNOT_chi.basis_r == (q2, q2)
@test CNOT_ptm.basis_l == CNOT_ptm.basis_r == (q2, q2)

@test all(isapprox.(imag.(CNOT_sop.data), 0))
@test all(isapprox.(imag.(CNOT_chi.data), 0))
@test all(isapprox.(imag.(CNOT_ptm.data), 0))

@test all(isapprox.(CNOT_sop.data[[1,18,36,51,69,86,104,119,141,158,176,191,201,218,236,251]], 1))
@test all(isapprox.(CNOT_chi.data[[1,2,13,17,18,29,193,194,205,222]], 1))
@test all(isapprox.(CNOT_chi.data[[14,30,206,209,210,221]], -1))
@test all(isapprox.(CNOT_ptm.data[[1,18,47,64,70,85,108,138,153,183,205,222,227,244,]], 1))
@test all(isapprox.(CNOT_ptm.data[[123,168]], -1))

# Test DenseChiMatrix constructor.
@test_throws DimensionMismatch DenseChiMatrix((q2, q2), (q3, q3), CNOT_chi.data)
@test DenseChiMatrix((q2, q2), (q2, q2), CNOT_chi.data) == CNOT_chi

# Test equality and conversion among all three bases.
ident = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

IDENT = DenseOperator(q2, ident)

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

CPHASE = DenseOperator(q2, cphase)

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
one_qubit_left_data = zeros(ComplexF64, 4, 4)
one_qubit_left_data[2, 2] = 2
one_qubit_right_data = zeros(ComplexF64, 4, 4)
one_qubit_right_data[3, 3] = 2
one_qubit_expected_data = zeros(ComplexF64, 4, 4)
one_qubit_expected_data[4, 4] = 2
one_qubit_left = DenseChiMatrix((b, b), (b, b), one_qubit_left_data)
one_qubit_right = DenseChiMatrix((b, b), (b, b), one_qubit_right_data)
one_qubit_expected = DenseChiMatrix((b, b), (b, b), one_qubit_expected_data)
one_qubit_left_before = copy(one_qubit_left.data)
one_qubit_right_before = copy(one_qubit_right.data)
one_qubit_composition = @inferred(one_qubit_left * one_qubit_right)
@test one_qubit_composition == one_qubit_expected
@test one_qubit_left.data == one_qubit_left_before
@test one_qubit_right.data == one_qubit_right_before

two_qubit_composition = @inferred(CPHASE_chi * CNOT_chi)
@test isapprox(two_qubit_composition, ChiMatrix(CPHASE * CNOT))
@test isapprox(PauliTransferMatrix(CPHASE) * PauliTransferMatrix(CNOT), PauliTransferMatrix(CPHASE * CNOT))

@testset "Three-qubit Chi-matrix composition" begin
    left_pauli = dense(sigmax(b)) ⊗ dense(sigmay(b)) ⊗ dense(sigmaz(b))
    right_pauli = dense(sigmaz(b)) ⊗ dense(identityoperator(b)) ⊗ dense(sigmax(b))
    left_chi = ChiMatrix(left_pauli)
    right_chi = ChiMatrix(right_pauli)
    @test count(x -> !iszero(x), left_chi.data) == 1
    @test count(x -> !iszero(x), right_chi.data) == 1

    composition = left_chi * right_chi
    @test size(composition.data) == (64, 64)
    @test composition.basis_l == left_chi.basis_l
    @test composition.basis_r == left_chi.basis_r
    @test isapprox(composition, ChiMatrix(left_pauli * right_pauli))
end

end # testset
end
