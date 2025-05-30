@testitem "jet" tags = [:jet] begin
    using QuantumOpticsBase
    using JET

    JET.test_package(QuantumOpticsBase, target_defined_modules = true)
end # testset
