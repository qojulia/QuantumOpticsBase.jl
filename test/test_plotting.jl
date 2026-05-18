@testitem "Bloch Sphere Plotting" tags=[:plotting] begin
    using QuantumOpticsBase
    using CairoMakie
    import Makie

    b = SpinBasis(1//2)

    # ── Return types ──────────────────────────────────────────────────────────
    @testset "blochsphere returns Figure, Axis3, and plot object" begin
        fig, ax, plt = blochsphere(spinup(b))
        @test fig isa Figure
        @test ax  isa Axis3
        @test plt isa Makie.Plot
    end

    # ── Render test ───────────────────────────────────────────────────────────
    @testset "arbitrary state renders without error" begin
        ψ = cos(π/6)*spinup(b) + exp(im*π/4)*sin(π/6)*spindown(b)
        fig, _, _ = blochsphere(ψ)
        save("test_bloch.png", fig)
        @test isfile("test_bloch.png")
        rm("test_bloch.png")
    end

    # ── Render tests: custom attributes ──────────────────────────────────────
    @testset "Custom arrowcolor and spherealpha" begin
        fig, _, _ = blochsphere(spinup(b); arrowcolor=:blue, spherealpha=0.3)
        save("test_bloch_custom_color.png", fig)
        @test isfile("test_bloch_custom_color.png")
        rm("test_bloch_custom_color.png")
    end

    @testset "Custom spherecolor" begin
        fig, _, _ = blochsphere(spindown(b); spherecolor=:pink, spherealpha=0.2)
        save("test_bloch_custom_sphere.png", fig)
        @test isfile("test_bloch_custom_sphere.png")
        rm("test_bloch_custom_sphere.png")
    end

    @testset "Wireframe, labels and axes toggled off" begin
        fig, _, _ = blochsphere(spinup(b); showwireframe=false, showlabels=false, showaxes=false)
        save("test_bloch_minimal.png", fig)
        @test isfile("test_bloch_minimal.png")
        rm("test_bloch_minimal.png")
    end

    # ── Error handling ────────────────────────────────────────────────────────
    @testset "Wrong dimension state throws error" begin
        # spin-1 has 3 levels — the recipe requires exactly 2
        b3  = SpinBasis(1)
        ψ_3 = basisstate(b3, 1)
        @test_throws ErrorException blochsphere(ψ_3)
    end
end
