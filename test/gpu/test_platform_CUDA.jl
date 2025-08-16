@testitem "CUDA" tags = [:cuda] begin

    include("implementation/test_platform.jl")

    using CUDA: CuArray, CUDA
    const AT = CuArray

    const can_run = CUDA.functional()

    @testset "Device availability" begin
        @test can_run
    end

    if can_run
        synchronize() = CUDA.synchronize()
        test_platform(AT, synchronize)
    else
        @info "Skipping CUDA tests - CUDA not functional"
    end

end
