using Test
using QuantumOpticsBase
using Aqua
using FillArrays

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase;
                  ambiguities=(exclude=[FillArrays.reshape],),  # Due to https://github.com/JuliaArrays/FillArrays.jl/issues/105#issuecomment-1518406018
                  piracy=(broken=true,)
                  )
    # manual piracy check to exclude identityoperator
    pirates = [pirate for pirate in Aqua.Piracy.hunt(QuantumOpticsBase) if pirate.name != :identityoperator]
    @test isempty(pirates)
end # testset
