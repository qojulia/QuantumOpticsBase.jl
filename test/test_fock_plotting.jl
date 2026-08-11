@testitem "Fock Distribution Plotting" tags=[:plotting] begin
    using QuantumOpticsBase
    using CairoMakie

    b = FockBasis(20)

    # ── Return types ──────────────────────────────────────────────────────────
    @testset "fockdistributionplot_axis returns Figure, Axis, and plot object" begin
        fig, ax, plt = fockdistributionplot_axis(coherentstate(b, 2.0))
        @test fig isa Figure
        @test ax  isa Axis
        @test plt isa AbstractPlot
    end

    # ── Render tests: kets ────────────────────────────────────────────────────
    @testset "coherent state renders without error" begin
        fig, _, _ = fockdistributionplot_axis(coherentstate(b, 2.0))
        save("test_fock.png", fig)
        @test isfile("test_fock.png")
        rm("test_fock.png")
    end

    @testset "number state renders without error" begin
        fig, _, _ = fockdistributionplot_axis(fockstate(b, 3))
        save("test_fock_number.png", fig)
        @test isfile("test_fock_number.png")
        rm("test_fock_number.png")
    end

    # ── Render test: density operator ─────────────────────────────────────────
    @testset "thermal density operator renders without error" begin
        ρ = thermalstate(number(b), 1.0)
        fig, _, _ = fockdistributionplot_axis(ρ)
        save("test_fock_thermal.png", fig)
        @test isfile("test_fock_thermal.png")
        rm("test_fock_thermal.png")
    end

    # ── Render tests: custom attributes ──────────────────────────────────────
    @testset "Custom color, alpha and width" begin
        fig, _, _ = fockdistributionplot_axis(coherentstate(b, 2.0); color=:purple, alpha=0.8, width=0.5)
        save("test_fock_custom.png", fig)
        @test isfile("test_fock_custom.png")
        rm("test_fock_custom.png")
    end

    # ── Observable reactivity ─────────────────────────────────────────────────
    @testset "Observable state updates reactively" begin
        using Makie: Observable
        state_obs = Observable(coherentstate(b, 1.0))
        fig, ax, _ = fockdistributionplot_axis(state_obs)
        state_obs[] = coherentstate(b, 3.0)
        save("test_fock_observable.png", fig)
        @test isfile("test_fock_observable.png")
        rm("test_fock_observable.png")
    end
end
