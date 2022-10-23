using Test
using QuantumOpticsBase
using Aqua

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase,
                  ambiguities=false,
                  unbound_args=false, # TODO due to Aqua bug https://github.com/JuliaTesting/Aqua.jl/issues/87
                  )
    @test_broken false # fix the ambiguities test above
    @test_broken false # fix the unbounds_args test above (it is an Aqua bug)
end # testset
