@testitem "test_jet" tags = [:jet] begin
using Test
using QuantumOpticsBase
using JET

@testset "jet" begin
    JET.test_package(QuantumOpticsBase, target_defined_modules = true)
end # testset
end
