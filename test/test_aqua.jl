using Test
using QuantumOpticsBase
using Aqua
using FillArrays

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase;
                  piracy=false,  # TODO: Due to Base methods in QuantumOpticsBase, for types defined in QuantumInterface
                  ambiguities=(exclude=[FillArrays.reshape],)  # Due to https://github.com/JuliaArrays/FillArrays.jl/issues/105#issuecomment-1518406018
                  )
end # testset
