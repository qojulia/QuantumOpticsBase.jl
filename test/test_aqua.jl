using Test
using QuantumOpticsBase
using Aqua

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase)
end # testset
