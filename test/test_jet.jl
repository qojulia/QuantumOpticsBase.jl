using Test
using QuantumOpticsBase
using JET

using JET: ReportPass, BasicPass, InferenceErrorReport, UncaughtExceptionReport

# Custom report pass that ignores `UncaughtExceptionReport`
# Too coarse currently, but it serves to ignore the various
# "may throw" messages for runtime errors we raise on purpose
# (mostly on malformed user input)
struct MayThrowIsOk <: ReportPass end

# ignores `UncaughtExceptionReport` analyzed by `JETAnalyzer`
(::MayThrowIsOk)(::Type{UncaughtExceptionReport}, @nospecialize(_...)) = return

# forward to `BasicPass` for everything else
function (::MayThrowIsOk)(report_type::Type{<:InferenceErrorReport}, @nospecialize(args...))
    BasicPass()(report_type, args...)
end

# imported to be declared as modules filtered out from analysis result
using LinearAlgebra, LRUCache, Strided, Dates, SparseArrays

@testset "jet" begin
    if get(ENV,"JET_TEST","")=="true"
        rep = report_package("QuantumOpticsBase";
            report_pass=MayThrowIsOk(), # TODO have something more fine grained than a generic "do not care about thrown errors"
            ignored_modules=( # TODO fix issues with these modules or report them upstream
                AnyFrameModule(LinearAlgebra),
                AnyFrameModule(LRUCache),
                AnyFrameModule(Strided),
                AnyFrameModule(Dates),
                AnyFrameModule(SparseArrays))
            )
        @show rep
        @test length(JET.get_reports(rep)) <= 1
        @test_broken length(JET.get_reports(rep)) == 0
    end
end # testset
