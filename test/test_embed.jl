@testitem "test_embed" begin
using Test
using QuantumOpticsBase
using Random, SparseArrays, LinearAlgebra

struct EmbedFallbackOperator{BL,BR,T} <: DataOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::T
end
Base.eltype(op::EmbedFallbackOperator) = eltype(op.data)

struct EmbedEquivalentBasisA <: Basis
    shape::Vector{Int}
end
struct EmbedEquivalentBasisB <: Basis
    shape::Vector{Int}
end
Base.:(==)(left::EmbedEquivalentBasisA, right::EmbedEquivalentBasisB) =
    left.shape == right.shape
Base.:(==)(left::EmbedEquivalentBasisB, right::EmbedEquivalentBasisA) = right == left

@testset "embed" begin

Random.seed!(0)

# Set up operators
spinbasis = SpinBasis(1//2)

b1 = NLevelBasis(3)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

I1 = dense(identityoperator(b1))
I2 = dense(identityoperator(b2))
I3 = dense(identityoperator(b3))

b = b1 ⊗ b2 ⊗ b3

op1 = DenseOperator(b1, b1, rand(ComplexF64, length(b1), length(b1)))
op2 = DenseOperator(b2, b2, rand(ComplexF64, length(b2), length(b2)))
op3 = DenseOperator(b3, b3, rand(ComplexF64, length(b3), length(b3)))


# Test Vector{Int}, Vector{AbstractOperator}
x = embed(b, [1,2], [op1, op2])
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, [1,2], [sparse(op1), sparse(op2)])
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, 1, op1)
y = op1 ⊗ I2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, 2, op2)
y = I1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, 3, op3)
y = I1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))


# Test Dict(Int=>AbstractOperator)
x = embed(b, Dict(1 => sparse(op1), 2 => sparse(op2)))
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, Dict(1 => op1, 2 => op2))
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, Dict([1,3] => sparse(op1⊗op3)))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, Dict([1,3] => op1⊗op3))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, Dict([3,1] => sparse(op3⊗op1)))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, Dict([3,1] => op3⊗op1))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(x, y))

@testset "inferred multi-index embedding" begin
    b1 = NLevelBasis(2)
    b2 = SpinBasis(1 // 2)
    b3 = FockBasis(2)
    b4 = GenericBasis(2)
    full_basis = b1 ⊗ b2 ⊗ b3 ⊗ b4

    op1_real = DenseOperator(b1, reshape(Float64.(1:4), 2, 2))
    op2_real = DenseOperator(b2, reshape(Float64.(1:4), 2, 2))
    op3_real = DenseOperator(b3, reshape(Float64.(1:9), 3, 3))
    op4_real = DenseOperator(b4, reshape(Float64.(1:4), 2, 2))
    op1_complex = DenseOperator(b1, (1 + 2im) .* op1_real.data)
    op3_complex = DenseOperator(b3, (2 - 1im) .* op3_real.data)

    expected_real = op1_real ⊗ identityoperator(b2) ⊗ op3_real ⊗ identityoperator(b4)
    selected_real = op1_real ⊗ op3_real
    indices = [1, 3]
    selected_data = copy(selected_real.data)
    embedded_real = @inferred embed(full_basis, indices, selected_real)
    @test embedded_real == expected_real
    @test embedded_real.data isa SparseMatrixCSC{Float64,Int}
    @test embedded_real.basis_l === full_basis
    @test embedded_real.basis_r === full_basis
    @test indices == [1, 3]
    @test selected_real.data == selected_data

    selected_complex = op1_complex ⊗ op3_complex
    expected_complex = op1_complex ⊗ identityoperator(b2) ⊗ op3_complex ⊗ identityoperator(b4)
    @test (@inferred embed(full_basis, [1, 3], selected_complex)) == expected_complex
    @test (@inferred embed(full_basis, [1, 3], sparse(selected_real))) == expected_real
    @test (@inferred embed(full_basis, [1, 3], sparse(selected_complex))) == expected_complex

    dense_adjoint = dagger(selected_complex)
    sparse_adjoint = dagger(sparse(selected_complex))
    expected_adjoint =
        dagger(op1_complex) ⊗ identityoperator(b2) ⊗ dagger(op3_complex) ⊗ identityoperator(b4)
    @test (@inferred embed(full_basis, [1, 3], dense_adjoint)) == expected_adjoint
    @test (@inferred embed(full_basis, [1, 3], sparse_adjoint)) == expected_adjoint

    selected_identity = identityoperator(b1 ⊗ b3)
    @test (@inferred embed(full_basis, [1, 3], selected_identity)) == identityoperator(full_basis)

    selected_reordered = op3_complex ⊗ op1_complex
    @test (@inferred embed(full_basis, [3, 1], selected_reordered)) == expected_complex

    expected_contiguous =
        identityoperator(b1) ⊗ op2_real ⊗ op3_real ⊗ identityoperator(b4)
    @test (@inferred embed(full_basis, [2, 3], op2_real ⊗ op3_real)) == expected_contiguous

    selected_permutation = op4_real ⊗ op2_real ⊗ op1_real ⊗ op3_real
    expected_permutation = op1_real ⊗ op2_real ⊗ op3_real ⊗ op4_real
    @test (@inferred embed(full_basis, [4, 2, 1, 3], selected_permutation)) == expected_permutation

    vector_result = @inferred embed(full_basis, [2], op2_real)
    integer_result = @inferred embed(full_basis, 2, op2_real)
    @test vector_result == integer_result

    @test_throws QuantumOpticsBase.IncompatibleBases embed(
        full_basis, [1, 3], op1_real ⊗ op2_real
    )
    @test_throws ArgumentError embed(full_basis, [1, 1], op1_real ⊗ op1_real)
    @test_throws ArgumentError embed(full_basis, [1, 5], op1_real ⊗ op1_real)
    @test_throws AssertionError embed(full_basis, b1 ⊗ b2 ⊗ b3, [1], op1_real)
end

@testset "rectangular multi-index embedding" begin
    dimensions_l = (2, 3, 4, 2)
    dimensions_r = (3, 2, 2, 4)
    bases_l = GenericBasis.(dimensions_l)
    bases_r = GenericBasis.(dimensions_r)
    basis_l = reduce(⊗, bases_l)
    basis_r = reduce(⊗, bases_r)

    op1 = DenseOperator(
        bases_l[1], bases_r[1], reshape(ComplexF64.(1:6), dimensions_l[1], dimensions_r[1])
    )
    op3 = DenseOperator(
        bases_l[3], bases_r[3], reshape(ComplexF64.(1:8), dimensions_l[3], dimensions_r[3])
    )
    selected = op1 ⊗ op3
    expected =
        op1 ⊗ identityoperator(bases_l[2], bases_r[2]) ⊗
        op3 ⊗ identityoperator(bases_l[4], bases_r[4])

    embedded = @inferred embed(basis_l, basis_r, [1, 3], selected)
    @test embedded == expected
    @test embedded.data isa SparseMatrixCSC{ComplexF64,Int}
end

@testset "DataOperator compatibility fallback" begin
    b1 = GenericBasis(2)
    b2 = GenericBasis(3)
    basis = b1 ⊗ b2
    selected_basis = b2 ⊗ b1
    data = reshape(ComplexF64.(1:36), 6, 6)
    custom = EmbedFallbackOperator(selected_basis, selected_basis, data)

    embedded = embed(basis, [2, 1], custom)
    reference = embed(basis, [2, 1], Operator(selected_basis, data))
    @test embedded isa EmbedFallbackOperator
    @test embedded.basis_l === basis
    @test embedded.basis_r === basis
    @test embedded.data == reference.data
    @test custom.data == data

    b3 = GenericBasis(2)
    full_basis = b1 ⊗ b2 ⊗ b3
    selected_basis = b1 ⊗ b3
    diagonal1 = Operator(b1, Diagonal([1.0, 2.0]))
    diagonal3 = Operator(b3, Diagonal([3.0, 4.0]))
    diagonal_operator = diagonal1 ⊗ diagonal3
    diagonal_data = diagonal_operator.data
    @test diagonal_data isa Diagonal
    @test embed(full_basis, [1, 3], diagonal_operator) ==
          diagonal1 ⊗ identityoperator(b2) ⊗ diagonal3
    @test diagonal_operator.data === diagonal_data
end

@testset "cross-type basis compatibility" begin
    basis_a = EmbedEquivalentBasisA([2])
    basis_b = EmbedEquivalentBasisB([2])
    gap_basis = GenericBasis(3)
    full_basis = basis_a ⊗ gap_basis
    data = [1.0 2.0; 3.0 4.0]
    selected = DenseOperator(basis_b, data)

    embedded = @inferred embed(full_basis, [1], selected)
    expected = DenseOperator(basis_a, data) ⊗ identityoperator(gap_basis)
    @test embedded == expected
end

end # testset
end
