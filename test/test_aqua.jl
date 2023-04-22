using Test
using QuantumOpticsBase
using Aqua

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase;
                  piracy=false,  # TODO: Due to Base methods in QuantumOpticsBase, for types defined in QuantumInterface
                  ambiguities=(exclude=[Base.reshape],)  # FIXME: Temporarily work-around ambiguities from FillArrays
                  )
end # testset
