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
