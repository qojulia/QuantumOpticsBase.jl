using Test
using QuantumOpticsBase
using Aqua
using FillArrays

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase;
                  ambiguities=(recursive=false),
                  piracy=(broken=true,)
                  )
    # manual piracy check to exclude identityoperator
    pirates = [pirate for pirate in Aqua.Piracy.hunt(QuantumOpticsBase) if pirate.name ∉ [:identityoperator,:identitysuperoperator]]
    @test isempty(pirates)
end # testset
