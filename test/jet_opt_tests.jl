@testitem "JET optimization: TimeDependentSum pairs" tags = [:jet] begin
using JET
using QuantumOpticsBase
using Test

basis = SpinBasis(1 // 2)
sx = sigmax(basis)
sz = sigmaz(basis)

JET.@test_opt target_modules=(QuantumOpticsBase,) TimeDependentSum(cos => sx, sin => sz)

operator = TimeDependentSum(cos => sx, sin => sz)
@test QuantumOpticsBase.coefficients(operator) == (cos, sin)
JET.@test_opt target_modules=(QuantumOpticsBase,) set_time!(operator, 0.25)
end

@testitem "LazyProduct constructor inference" tags = [:jet] begin
using Test
using QuantumOpticsBase
using JET
using LinearAlgebra

basis_1 = SpinBasis(1 // 2)
basis_2 = FockBasis(1)
basis_3 = NLevelBasis(2)
basis_4 = GenericBasis(2)
data = Matrix{ComplexF64}(I, 2, 2)

operator_1 = DenseOperator(basis_1, basis_2, data)
operator_2 = DenseOperator(basis_2, basis_3, data)
operator_3 = DenseOperator(basis_3, basis_4, data)
operator_4 = DenseOperator(basis_4, basis_1, data)

JET.@test_opt target_modules=(QuantumOpticsBase,) LazyProduct(
    operator_1,
    operator_2,
    operator_3,
    operator_4,
)
end

@testitem "JET optimization: partial transpose metrics" tags = [:jet] begin
using Test
using QuantumOpticsBase
using JET

basis = SpinBasis(1 // 2)
up = spinup(basis)
down = spindown(basis)
state = (up ⊗ down - down ⊗ up) / sqrt(2)
rho = dm(state)

JET.@test_opt target_modules=(QuantumOpticsBase,) ptranspose(rho, [1])
JET.@test_opt target_modules=(QuantumOpticsBase,) negativity(rho, 1)
end

@testitem "JET optimization: fundamental quantum operations" tags = [:jet] begin
using Test
using JET
using QuantumOpticsBase
using SparseArrays: sparse
using LinearAlgebra: mul!

spin_basis = SpinBasis(1 // 2)
fock_basis = FockBasis(3)

spin_up = basisstate(spin_basis, 1)
spin_down = basisstate(spin_basis, 2)
fock_one_photon = basisstate(fock_basis, 2)
spin_superposition = spin_up + spin_down
composite_basis = spin_basis ⊗ fock_basis
composite_state = spin_up ⊗ fock_one_photon
spin_density = dm(spin_up)
fock_density = dm(fock_one_photon)
composite_density = dm(composite_state)

sx_sparse = sigmax(spin_basis)
sz_sparse = sigmaz(spin_basis)
sx_dense = dense(sx_sparse)
sz_dense = dense(sz_sparse)
fock_number = number(fock_basis)
spin_identity = identityoperator(spin_basis)
mul_output = copy(spin_up)

jump_operators = [destroy(fock_basis)]
fock_liouvillian = liouvillian(fock_number, jump_operators)

@testset "State algebra" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(spin_basis, 1)
    JET.@test_opt target_modules=(QuantumOpticsBase,) dagger(spin_up)
    JET.@test_opt target_modules=(QuantumOpticsBase,) spin_up + spin_down
    JET.@test_opt target_modules=(QuantumOpticsBase,) normalize(spin_superposition)
    JET.@test_opt target_modules=(QuantumOpticsBase,) dagger(spin_up) * spin_up
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(spin_up, fock_one_photon)
    JET.@test_opt target_modules=(QuantumOpticsBase,) dm(spin_up)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tr(spin_density)
end

@testset "Operator kernels" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) dense(sx_sparse)
    JET.@test_opt target_modules=(QuantumOpticsBase,) sparse(sx_dense)
    JET.@test_opt target_modules=(QuantumOpticsBase,) dagger(sx_dense)
    JET.@test_opt target_modules=(QuantumOpticsBase,) sx_sparse + sz_sparse
    JET.@test_opt target_modules=(QuantumOpticsBase,) sx_dense * spin_up
    JET.@test_opt target_modules=(QuantumOpticsBase,) sx_sparse * spin_up
    JET.@test_opt target_modules=(QuantumOpticsBase,) sx_dense * sz_sparse
    JET.@test_opt target_modules=(QuantumOpticsBase,) sx_sparse * sz_dense
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(sx_sparse, fock_number)
    JET.@test_opt target_modules=(QuantumOpticsBase,) mul!(
        mul_output, sx_sparse, spin_down, 1.0 + 0im, 0.0 + 0im
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) spin_identity * spin_up
end

@testset "Observables and composite systems" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(sx_dense, spin_up)
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(sz_sparse, spin_density)
    JET.@test_opt target_modules=(QuantumOpticsBase,) variance(sz_sparse, spin_up)
    JET.@test_opt target_modules=(QuantumOpticsBase,) embed(composite_basis, 1, sx_sparse)
    JET.@test_opt target_modules=(QuantumOpticsBase,) apply!(
        copy(composite_state), 1, sx_sparse
    )
end

@testset "Open systems" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) liouvillian(
        fock_number, jump_operators
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) fock_liouvillian * fock_density
end

@testset "Inference-sensitive composite operations" begin
    @testset "Composite state construction" begin
        JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
            composite_basis, [1, 1]
        )
        JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
            spin_up, fock_one_photon, spin_down
        )
    end

    @testset "Partial trace and indexed expectation" begin
        JET.@test_opt target_modules=(QuantumOpticsBase,) ptrace(
            composite_state, 1
        )
        JET.@test_opt target_modules=(QuantumOpticsBase,) ptrace(
            composite_density, 1
        )
        JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
            1, sx_sparse, composite_state
        )
    end

    @testset "System permutations" begin
        JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
            composite_state, [2, 1]
        )
        JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
            composite_density, [2, 1]
        )
    end
end
end

@testitem "JET optimization: generalized state construction" tags = [:jet] begin
using Test
using JET
using QuantumOpticsBase

basis_1 = SpinBasis(1 // 2)
basis_2 = FockBasis(3)
basis_3 = NLevelBasis(3)
basis_4 = GenericBasis(5)
basis_5 = GenericBasis(2)
basis_123 = tensor(basis_1, basis_2, basis_3)
basis_1234 = tensor(basis_1, basis_2, basis_3, basis_4)
shaped_basis = GenericBasis([2, 3, 4, 5])

@testset "Basis states" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
        basis_1234, length(basis_1234)
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
        basis_123, [1, 2, 3]
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
        ComplexF32, basis_1234, (1, 2, 3, 4)
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
        basis_1234, (Int8(1), UInt8(2), Int32(3), Int64(4))
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
        basis_1234, 1:4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) basisstate(
        shaped_basis, [1, 2, 3, 4]
    )

    JET.@test_opt target_modules=(QuantumOpticsBase,) sparsebasisstate(
        basis_123, (1, 2, 3)
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) sparsebasisstate(
        Float32, basis_1234, [1, 2, 3, 4]
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) sparsebasisstate(
        shaped_basis, (1, 2, 3, 4)
    )

    # Abstractly typed coordinates still dispatch dynamically in the stride loop.
    abstract_coordinates = Integer[1, 2, 3, 4]
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) basisstate(
        basis_1234, abstract_coordinates
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) sparsebasisstate(
        basis_1234, abstract_coordinates
    )
end

ket_1 = basisstate(Float32, basis_1, 1)
ket_2 = sparsebasisstate(ComplexF64, basis_2, 2)
ket_3 = basisstate(Float64, basis_3, 3)
ket_4 = sparsebasisstate(ComplexF32, basis_4, 4)
ket_5 = basisstate(Int, basis_5, 2)
bra_1, bra_2, bra_3, bra_4, bra_5 = dagger.((ket_1, ket_2, ket_3, ket_4, ket_5))
dense_ket_2 = basisstate(ComplexF64, basis_2, 2)
dense_ket_4 = basisstate(ComplexF32, basis_4, 4)
dense_bra_2 = dagger(dense_ket_2)
dense_bra_4 = dagger(dense_ket_4)
sparse_ket_1 = sparsebasisstate(Float32, basis_1, 1)
sparse_ket_3 = sparsebasisstate(Float64, basis_3, 3)
sparse_ket_5 = sparsebasisstate(Int, basis_5, 2)
sparse_bra_1, sparse_bra_3, sparse_bra_5 =
    dagger.((sparse_ket_1, sparse_ket_3, sparse_ket_5))

@testset "State tensors" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(ket_1)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(ket_2)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(bra_1)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(bra_2)

    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(ket_1, ket_2)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(ket_2, ket_3)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(bra_1, bra_2)
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(bra_2, bra_3)

    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        ket_1, ket_2, ket_3, ket_4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        bra_1, bra_2, bra_3, bra_4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        ket_1, ket_2, ket_3, ket_4, ket_5
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        bra_1, bra_2, bra_3, bra_4, bra_5
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        ket_1, dense_ket_2, ket_3, dense_ket_4, ket_5
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        bra_1, dense_bra_2, bra_3, dense_bra_4, bra_5
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        sparse_ket_1, ket_2, sparse_ket_3, ket_4, sparse_ket_5
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        sparse_bra_1, bra_2, sparse_bra_3, bra_4, sparse_bra_5
    )

    composite_ket_12 = tensor(ket_1, ket_2)
    composite_ket_23 = tensor(ket_2, ket_3)
    composite_ket_45 = tensor(ket_4, ket_5)
    composite_bra_12 = dagger(composite_ket_12)
    composite_bra_23 = dagger(composite_ket_23)
    composite_bra_45 = dagger(composite_ket_45)

    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        composite_ket_12, ket_3, ket_4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        ket_1, composite_ket_23, ket_4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        ket_1, ket_3, composite_ket_45
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        composite_ket_12, composite_ket_45, ket_3
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        composite_bra_12, bra_3, bra_4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        bra_1, composite_bra_23, bra_4
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        bra_1, bra_3, composite_bra_45
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) tensor(
        composite_bra_12, composite_bra_45, bra_3
    )
end
end

@testitem "JET optimization: generalized traces and observables" tags = [:jet] begin
using Test
using JET
using QuantumOpticsBase
using SparseArrays: sparse

basis_1 = SpinBasis(1 // 2)
basis_2 = FockBasis(1)
basis_3 = NLevelBasis(3)
basis_4 = GenericBasis(2)
ket_1 = basisstate(basis_1, 1)
ket_2 = basisstate(basis_2, 2)
ket_3 = basisstate(basis_3, 3)
ket_4 = basisstate(basis_4, 1)

state_12 = tensor(ket_1, ket_2)
bra_12 = dagger(state_12)
density_12 = dm(state_12)
state_123 = tensor(ket_1, ket_2, ket_3)
bra_123 = dagger(state_123)
density_123 = dm(state_123)
state_1234 = tensor(ket_1, ket_2, ket_3, ket_4)
bra_1234 = dagger(state_1234)
density_1234 = dm(state_1234)

left_12 = basis_1 ⊗ basis_2
right_12 = GenericBasis(2) ⊗ NLevelBasis(3)
dense_rectangular_12 = DenseOperator(
    left_12,
    right_12,
    reshape(ComplexF32.(1:24), 4, 6),
)
sparse_rectangular_12 = sparse(dense_rectangular_12)

left_123 = basis_1 ⊗ basis_2 ⊗ NLevelBasis(4)
right_123 = GenericBasis(2) ⊗ NLevelBasis(3) ⊗ FockBasis(3)
dense_rectangular_123 = DenseOperator(
    left_123,
    right_123,
    reshape(ComplexF32.(1:384), 16, 24),
)
sparse_rectangular_123 = sparse(dense_rectangular_123)

left_1234 = basis_1 ⊗ basis_2 ⊗ basis_3 ⊗ FockBasis(3)
right_1234 = GenericBasis(2) ⊗ NLevelBasis(3) ⊗ FockBasis(2) ⊗ NLevelBasis(5)
dense_rectangular_1234 = DenseOperator(
    left_1234,
    right_1234,
    reshape(ComplexF64.(1:4320), 48, 90),
)
sparse_rectangular_1234 = sparse(dense_rectangular_1234)

operator_1_dense = DenseOperator(basis_1, ComplexF64[0 1; 2im 0])
operator_1_sparse = sparse(operator_1_dense)
operator_2_dense = DenseOperator(basis_2, Float64[0 2; -1 0])
operator_2_sparse = sparse(operator_2_dense)
operator_3_dense = DenseOperator(basis_3, Float32[0 1 0; 0 0 2; 3 0 0])
operator_3_sparse = sparse(operator_3_dense)
operator_12_dense = tensor(operator_1_dense, operator_2_dense)
operator_12_sparse = sparse(operator_12_dense)
state_123_f32 = Ket(state_123.basis, ComplexF32.(state_123.data))
state_123_sparse = Ket(state_123.basis, sparse(state_123.data))
homogeneous_state_1234 = tensor(ket_1, ket_1, ket_1, ket_1)

# A broken marker must emit at least one package-owned report. If the call becomes
# inference-clean, the test fails so that the marker can be removed.
@testset "Partial traces and reductions" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) ptrace(bra_12, 1)
    JET.@test_opt target_modules=(QuantumOpticsBase,) ptrace(
        dense_rectangular_12, 1
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        state_12, [1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        bra_12, [2]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        density_12, [1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        state_12, 1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        density_12, [2]
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        state_123, 1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        state_123, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        bra_123, [2]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        density_123, 3
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        density_123, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        state_123, 2
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        bra_123, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        density_123, [2]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        state_123_sparse, 1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        dagger(state_123_sparse), [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        state_123_sparse, [2]
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        state_1234, 2
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        state_1234, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        bra_1234, [1, 3, 4]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        density_1234, 4
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        density_1234, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) reduced(
        density_1234, [1, 3]
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        sparse_rectangular_12, 1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        dense_rectangular_123, 1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        dense_rectangular_123, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        sparse_rectangular_123, [1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        sparse_rectangular_123, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        dense_rectangular_1234, [1, 3]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) ptrace(
        sparse_rectangular_1234, [1, 3]
    )
end

@testset "Indexed expectations and variances" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
        1, operator_1_sparse, state_123
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
        2, operator_2_sparse, state_123
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
        3, operator_3_sparse, state_123
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
        1, operator_1_sparse, state_123_f32
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
        1, operator_1_sparse, state_123_sparse
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) expect(
        2, operator_1_sparse, homogeneous_state_1234
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        1, operator_1_dense, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        [1], operator_1_sparse, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        1, dagger(operator_1_sparse), state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        [1, 2], operator_12_dense, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        [1, 2], operator_12_sparse, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        2, operator_2_sparse, state_1234
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        1, operator_1_dense, density_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        2, operator_2_sparse, density_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        [1, 2], operator_12_sparse, density_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) expect(
        2, operator_2_sparse, density_1234
    )

    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) variance(
        1, operator_1_dense, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) variance(
        2, operator_2_sparse, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) variance(
        [1, 2], operator_12_dense, state_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) variance(
        2, operator_2_sparse, state_1234
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) variance(
        1, operator_1_dense, density_123
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) variance(
        2, operator_2_sparse, density_1234
    )
end
end

@testitem "JET optimization: generalized system permutations" tags = [:jet] begin
using Test
using JET
using QuantumOpticsBase
using SparseArrays: sparse

basis_1 = SpinBasis(1 // 2)
basis_2 = FockBasis(2)
basis_3 = NLevelBasis(2)
basis_4 = GenericBasis(2)

basis_12 = basis_1 ⊗ basis_2
basis_123 = basis_12 ⊗ basis_3
basis_1234 = basis_123 ⊗ basis_4
homogeneous_basis_123 = basis_1 ⊗ basis_1 ⊗ basis_1

state_12 = basisstate(basis_12, [1, 2])
state_123 = basisstate(basis_123, [1, 2, 2])
state_1234 = basisstate(basis_1234, [1, 2, 2, 1])
homogeneous_state_123 = basisstate(homogeneous_basis_123, [1, 2, 1])
sparse_state_12 = sparsebasisstate(basis_12, [1, 2])
sparse_state_123 = sparsebasisstate(basis_123, [1, 2, 2])

right_basis_12 = GenericBasis(4) ⊗ NLevelBasis(2)
right_basis_123 = right_basis_12 ⊗ FockBasis(1)
right_basis_1234 = right_basis_123 ⊗ GenericBasis(2)
dense_rectangular_12 = DenseOperator(
    basis_12,
    right_basis_12,
    zeros(ComplexF64, length(basis_12), length(right_basis_12)),
)
dense_rectangular_123 = DenseOperator(
    basis_123,
    right_basis_123,
    zeros(ComplexF64, length(basis_123), length(right_basis_123)),
)
dense_rectangular_1234 = DenseOperator(
    basis_1234,
    right_basis_1234,
    zeros(ComplexF64, length(basis_1234), length(right_basis_1234)),
)
sparse_rectangular_12 = sparse(dense_rectangular_12)
sparse_rectangular_123 = sparse(dense_rectangular_123)
sparse_rectangular_1234 = sparse(dense_rectangular_1234)

dense_identity_12 = dense(identityoperator(basis_12))
dense_identity_123 = dense(identityoperator(basis_123))
lazy_tensor_12 = LazyTensor(basis_12, (1,), (sigmax(basis_1),))
lazy_tensor_123 = LazyTensor(
    basis_123,
    (1, 3),
    (sigmax(basis_1), identityoperator(basis_3)),
)
lazy_tensor_1234 = LazyTensor(
    basis_1234,
    (1, 4),
    (sigmax(basis_1), identityoperator(basis_4)),
)
lazy_sum_12 = LazySum(dense_identity_12, 2 * dense_identity_12)
lazy_sum_123 = LazySum(dense_identity_123, 2 * dense_identity_123)
lazy_product_12 = LazyProduct(dense_identity_12, dense_identity_12)
lazy_product_123 = LazyProduct(dense_identity_123, dense_identity_123)

@testset "Bipartite fast paths" begin
    JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
        dagger(state_12), [2, 1]
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
        sparse_state_12, Int8[2, 1]
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
        state_12, 2:-1:1
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
        dense_rectangular_12, view([2, 1], :)
    )
    JET.@test_opt target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_sum_12, [2, 1]
    )
end

@testset "General-rank state debt" begin
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        state_123, [2, 3, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dagger(state_123), 3:-1:1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dm(state_123), view([1, 2, 3], :)
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        homogeneous_state_123, [2, 3, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        sparse_state_123, [2, 3, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        state_1234, Int8[2, 3, 4, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dagger(state_1234), [4, 3, 2, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dm(state_1234), 1:4
    )
end

@testset "Operator storage and wrapper debt" begin
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dense_rectangular_123, [2, 3, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dense_rectangular_1234, [2, 3, 4, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        sparse_rectangular_12, [2, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        sparse_rectangular_123, 3:-1:1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        sparse_rectangular_1234, 4:-1:1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        identityoperator(basis_123), [2, 3, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dagger(dense_rectangular_12), [2, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        dagger(dense_rectangular_123), [2, 3, 1]
    )
end

@testset "Lazy operator debt" begin
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_tensor_12, [2, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_tensor_123, 3:-1:1
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_tensor_1234, [2, 3, 4, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_sum_123, [2, 3, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_product_12, [2, 1]
    )
    JET.@test_opt broken=true target_modules=(QuantumOpticsBase,) permutesystems(
        lazy_product_123, [2, 3, 1]
    )
end
end
