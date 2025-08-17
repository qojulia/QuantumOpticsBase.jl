include("imports.jl")
include("definitions.jl")
include("utilities.jl")
include("test_basic_operations.jl")
include("test_adapt_methods.jl")

@inline function test_platform(AT, synchronize)
    @testset "Basic Operations" begin
        test_basic_operations(AT, synchronize)
    end
    
    @testset "Adapt Methods" begin
        test_adapt_methods(AT, synchronize)
    end
end
