using Test
using QuantumOpticsBase
using JET

# imported to be declared as modules filtered out from analysis result
using LinearAlgebra, LRUCache, Strided, StridedViews, Dates, SparseArrays, RandomMatrices

@testset "jet" begin
    if get(ENV,"JET_TEST","")=="true"
        rep = report_package("QuantumOpticsBase";
            ignored_modules=( # TODO fix issues with these modules or report them upstream
                AnyFrameModule(LinearAlgebra),
                AnyFrameModule(LRUCache),
                AnyFrameModule(Strided),
                AnyFrameModule(StridedViews),
                AnyFrameModule(Dates),
                AnyFrameModule(SparseArrays),
                AnyFrameModule(RandomMatrices))
            )
        @show rep
        @test length(JET.get_reports(rep)) <= 24
        @test_broken length(JET.get_reports(rep)) == 0
    end
end # testset
