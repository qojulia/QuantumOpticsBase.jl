using QuantumOpticsBase
using Aqua
using FillArrays
using StatsBase

@testitem "aqua" tags = [:aqua] begin
    Aqua.test_all(QuantumOpticsBase;
                  ambiguities=(exclude=[FillArrays.reshape, # Due to https://github.com/JuliaArrays/FillArrays.jl/issues/105#issuecomment-1518406018
                                        StatsBase.TestStat, StatsBase.:(==) , StatsBase.sort!],),  # Due to https://github.com/JuliaStats/StatsBase.jl/issues/861
                  piracies=(broken=true,)
                  )
    # manual piracy check to exclude identityoperator
    pirates = [pirate for pirate in Aqua.Piracy.hunt(QuantumOpticsBase) if pirate.name ∉ [:identityoperator,:identitysuperoperator]]
    @test isempty(pirates)
end # testset
