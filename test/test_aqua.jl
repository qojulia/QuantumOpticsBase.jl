using Test
using QuantumOpticsBase
using Aqua

@testset "aqua" begin
    Aqua.test_all(QuantumOpticsBase,
                  piracy=false,  # TODO: Due to Base methods in QuantumOpticsBase, for types defined in QuantumInterface
                  ambiguities=false,  # FIXME: TEMPORARILY WORK AROUND AMBIGUITIES IN FILLARRAYS
                  )
end # testset
