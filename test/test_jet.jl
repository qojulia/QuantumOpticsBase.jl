using Test
using QuantumOpticsBase
using JET

@testset "jet" begin
    if get(ENV,"QUANTUMOPTICS_JET_TEST","")=="true"
        rep = report_package("QuantumOpticsBase"; ignored_modules=())
        @show rep
        @test_broken length(JET.get_reports(rep)) == 0
    end
end # testset
