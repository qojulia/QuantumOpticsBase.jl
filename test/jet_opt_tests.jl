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
