@testitem "test_aqua" tags = [:aqua] begin
using Test
using QuantumOpticsBase
using Aqua
using FillArrays
using StatsBase

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase;
                  ambiguities=(exclude=[FillArrays.reshape, # Due to https://github.com/JuliaArrays/FillArrays.jl/issues/105#issuecomment-1518406018
                                        StatsBase.TestStat, StatsBase.:(==) , StatsBase.sort!],),  # Due to https://github.com/JuliaStats/StatsBase.jl/issues/861
                  piracies=(broken=true,)
                  )
    # manual piracy check to exclude QuantumInterface functions that construct QuantumOpticsBase operators (as QOB.jl takes priority over other users of QuantumInterface)
    pirates = [pirate for pirate in Aqua.Piracy.hunt(QuantumOpticsBase) if pirate.name ∉ [:identityoperator,:identitysuperoperator, :position]]
    @test isempty(pirates)
end # testset
end
