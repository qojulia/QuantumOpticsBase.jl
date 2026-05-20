@testitem "Bloch Sphere Plotting" tags=[:plotting] begin
    using QuantumOpticsBase
    using CairoMakie   

    b = SpinBasis(1//2)

    # ── Return types ──────────────────────────────────────────────────────────
    @testset "blochsphereplot_axis returns Figure, Axis3, and plot object" begin
        fig, ax, plt = blochsphereplot_axis(spinup(b))
        @test fig isa Figure
        @test ax  isa Axis3
        @test plt isa AbstractPlot
    end

    # ── Render test ───────────────────────────────────────────────────────────
    @testset "arbitrary state renders without error" begin
        ψ = cos(π/6)*spinup(b) + exp(im*π/4)*sin(π/6)*spindown(b)
        fig, _, _ = blochsphereplot_axis(ψ)
        save("test_bloch.png", fig)
        @test isfile("test_bloch.png")
        rm("test_bloch.png")
    end

    # ── Render tests: custom attributes ──────────────────────────────────────
    @testset "Custom arrowcolor and spherealpha" begin
        fig, _, _ = blochsphereplot_axis(spinup(b); arrowcolor=:blue, spherealpha=0.3)
        save("test_bloch_custom_color.png", fig)
        @test isfile("test_bloch_custom_color.png")
        rm("test_bloch_custom_color.png")
    end

    @testset "Custom spherecolor" begin
        fig, _, _ = blochsphereplot_axis(spindown(b); spherecolor=:pink, spherealpha=0.2)
        save("test_bloch_custom_sphere.png", fig)
        @test isfile("test_bloch_custom_sphere.png")
        rm("test_bloch_custom_sphere.png")
    end

    @testset "Wireframe, labels and axes toggled off" begin
        fig, _, _ = blochsphereplot_axis(spinup(b); showwireframe=false, showlabels=false, showaxes=false)
        save("test_bloch_minimal.png", fig)
        @test isfile("test_bloch_minimal.png")
        rm("test_bloch_minimal.png")
    end

    # ── Error handling ────────────────────────────────────────────────────────
    @testset "Wrong dimension state throws error" begin
        b3  = SpinBasis(1)
        ψ_3 = basisstate(b3, 1)
        @test_throws ErrorException blochsphereplot_axis(ψ_3)
    end

    # ── Observable reactivity ─────────────────────────────────────────────────
    @testset "Observable state updates reactively" begin
        using Makie: Observable
        state_obs = Observable(spinup(b))
        fig = Figure(size = (700, 700))
        ax  = Axis3(fig[1, 1])
        blochsphereplot!(ax, state_obs)
        state_obs[] = spindown(b)
        save("test_bloch_observable.png", fig)
        @test isfile("test_bloch_observable.png")
        rm("test_bloch_observable.png")
    end
end
