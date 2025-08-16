include("imports.jl")
include("definitions.jl")
include("utilities.jl")
include("test_basic_operations.jl")

@inline function test_platform(AT, synchronize)
    @testset "Basic Operations" begin
        test_basic_operations(AT, synchronize)
    end
end