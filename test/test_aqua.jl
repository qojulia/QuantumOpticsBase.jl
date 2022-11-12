using Test
using QuantumOpticsBase
using Aqua

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase,
                  ambiguities=false,
                  )
    @test_broken false # fix the ambiguities test above
end # testset
