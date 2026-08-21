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
