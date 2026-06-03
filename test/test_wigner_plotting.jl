@testitem "Wigner Plot" tags=[:plotting] begin
    using QuantumOpticsBase
    using CairoMakie
 
    b = FockBasis(10)   # truncated Fock space — adequate for coherent α≤2, Fock n≤3
 
    # ── Return types ──────────────────────────────────────────────────────────
    @testset "wignerplot_axis returns Figure, Axis, and plot object" begin
        ψ = coherentstate(b, 1.0)
        fig, ax, plt = wignerplot_axis(ψ)
        @test fig isa Figure
        @test ax  isa Axis3
        @test plt isa AbstractPlot
    end
 
    # ── Render tests ──────────────────────────────────────────────────────────
    @testset "coherent state renders without error" begin
        ψ = coherentstate(b, 2.0)
        fig, _, _ = wignerplot_axis(ψ)
        save("test_wigner_coherent.png", fig)
        @test isfile("test_wigner_coherent.png")
        rm("test_wigner_coherent.png")
    end
 
    @testset "Fock state renders without error (Wigner can go negative)" begin
        ψ = fockstate(b, 3)
        fig, _, _ = wignerplot_axis(ψ)
        save("test_wigner_fock.png", fig)
        @test isfile("test_wigner_fock.png")
        rm("test_wigner_fock.png")
    end
 
    # ── Custom attributes ─────────────────────────────────────────────────────
    @testset "Custom xrange, prange, npoints" begin
        ψ = coherentstate(b, 0.5)
        fig, _, _ = wignerplot_axis(ψ; xrange=(-3.0, 3.0), prange=(-3.0, 3.0), npoints=50)
        save("test_wigner_custom_range.png", fig)
        @test isfile("test_wigner_custom_range.png")
        rm("test_wigner_custom_range.png")
    end
 
    @testset "Custom colormap" begin
        ψ = coherentstate(b, 1.0)
        fig, _, _ = wignerplot_axis(ψ; colormap=:bwr)
        save("test_wigner_custom_colormap.png", fig)
        @test isfile("test_wigner_custom_colormap.png")
        rm("test_wigner_custom_colormap.png")
    end
 
    # ── Observable reactivity ─────────────────────────────────────────────────
    @testset "Observable state updates reactively" begin
        using Makie: Observable
        state_obs = Observable(coherentstate(b, 1.0))
        fig, ax, _ = wignerplot_axis(state_obs)
        state_obs[] = coherentstate(b, -1.0)   # shift coherent peak to opposite side
        save("test_wigner_observable.png", fig)
        @test isfile("test_wigner_observable.png")
        rm("test_wigner_observable.png")
    end
 
    # ── Error handling ────────────────────────────────────────────────────────
    @testset "Wrong basis type throws error" begin
        # wigner is only defined for FockBasis states
        b_spin = SpinBasis(1//2)
        ψ_spin = spinup(b_spin)
        @test_throws Exception wignerplot_axis(ψ_spin)
    end
end