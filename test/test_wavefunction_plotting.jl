@testitem "Wave Function Plot" tags=[:plotting] begin
    using QuantumOpticsBase
    using QuantumOptics
    using CairoMakie

    b = PositionBasis(-5, 5, 128)   # 128 points is plenty for a σ=1 gaussian

    # ── Return types ──────────────────────────────────────────────────────────
    @testset "wavefunctionplot_axis returns Figure, Axis, and plot object" begin
        ψ = gaussianstate(b, 0.0, 0.0, 1.0)
        fig, ax, plt = wavefunctionplot_axis(ψ)
        @test fig isa Figure
        @test ax  isa Axis
        @test plt isa AbstractPlot
    end

    # ── Render tests ──────────────────────────────────────────────────────────
    @testset "gaussian state in position basis renders without error" begin
        ψ = gaussianstate(b, 1.0, 0.0, 1.0)
        fig, _, _ = wavefunctionplot_axis(ψ)
        save("test_wavefunction_position.png", fig)
        @test isfile("test_wavefunction_position.png")
        rm("test_wavefunction_position.png")
    end

    @testset "gaussian state in momentum basis renders without error" begin
        bp = MomentumBasis(b)
        ψ  = gaussianstate(bp, 0.0, 1.0, 1.0)
        fig, _, _ = wavefunctionplot_axis(ψ)
        save("test_wavefunction_momentum.png", fig)
        @test isfile("test_wavefunction_momentum.png")
        rm("test_wavefunction_momentum.png")
    end

    # ── Custom attributes ─────────────────────────────────────────────────────
    @testset "Each part renders" begin
        ψ = gaussianstate(b, 0.0, 2.0, 1.0)   # p0≠0 gives a non-trivial complex phase
        for part in (:abs2, :abs, :real, :imag)
            fig, _, _ = wavefunctionplot_axis(ψ; part=part)
            save("test_wavefunction_$part.png", fig)
            @test isfile("test_wavefunction_$part.png")
            rm("test_wavefunction_$part.png")
        end
    end

    @testset "Overlaying parts on one axis" begin
        ψ = gaussianstate(b, 0.0, 2.0, 1.0)
        fig, ax, _ = wavefunctionplot_axis(ψ; part=:real)
        wavefunctionplot!(ax, ψ; part=:imag)
        save("test_wavefunction_overlay.png", fig)
        @test isfile("test_wavefunction_overlay.png")
        rm("test_wavefunction_overlay.png")
    end

    # ── Observable reactivity ─────────────────────────────────────────────────
    @testset "Observable state updates reactively" begin
        using Makie: Observable
        state_obs = Observable(gaussianstate(b, -2.0, 0.0, 1.0))
        fig, ax, _ = wavefunctionplot_axis(state_obs)
        state_obs[] = gaussianstate(b, 2.0, 0.0, 1.0)   # shift the packet across the origin
        save("test_wavefunction_observable.png", fig)
        @test isfile("test_wavefunction_observable.png")
        rm("test_wavefunction_observable.png")
    end

    # ── Error handling ────────────────────────────────────────────────────────
    @testset "Wrong basis type throws error" begin
        # a wave function is only defined for particle states
        b_spin = SpinBasis(1//2)
        ψ_spin = spinup(b_spin)
        @test_throws "PositionBasis" wavefunctionplot_axis(ψ_spin)
    end

    @testset "Mixed state throws error" begin
        # a density operator has no wave function
        ρ = dm(gaussianstate(b, 0.0, 0.0, 1.0))
        @test_throws "Ket" wavefunctionplot_axis(ρ)
    end

    @testset "Unknown part throws error" begin
        ψ = gaussianstate(b, 0.0, 0.0, 1.0)
        @test_throws "part" wavefunctionplot_axis(ψ; part=:nonsense)
    end
end
