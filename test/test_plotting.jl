@testitem "Quantum state plotting" tags=[:plotting] begin
    using QuantumOptics
    using CairoMakie
    using LinearAlgebra

    @testset "Bloch vectors agree with Pauli expectations" begin
        b = SpinBasis(1//2)
        up, down = spinup(b), spindown(b)
        states = [up, down, normalize(up + down), normalize(up - down),
            normalize(up + im*down), normalize(up - im*down)]
        fig = Figure()
        ax = Axis3(fig[1, 1])
        for state in vcat(states, dm.(states), [0.5dm(up) + 0.5dm(down)])
            plot = blochsphereplot!(ax, state)
            expected = real.([expect(op, state) for op in (sigmax(b), sigmay(b), sigmaz(b))])
            @test iszero(expected) ? isempty(plot.directions[]) : isapprox(only(plot.directions[]), expected; atol=1e-7)
        end
        state = Observable(up)
        fig, ax, plot = blochsphereplot(state)
        @test ax isa Axis3
        state[] = down
        @test only(plot.directions[]) ≈ [0, 0, -1]
        surface, wireframe, arrow = plot.plots
        @test surface isa Makie.Mesh
        @test wireframe isa Makie.Lines
        @test count(p -> any(isnan, p), wireframe[1][]) == 17
        @test all(p -> any(isnan, p) || norm(p) ≈ 1, wireframe[1][])
        Makie.update!(plot; sphereresolution=(8, 4), spherecolor=(:gray, 0.25),
            wireframecolor=:red, wireframewidth=2)
        @test count(p -> any(isnan, p), wireframe[1][]) == 11
        @test Makie.to_color(surface.color[]) == Makie.to_color((:gray, 0.25))
        @test Makie.to_color(wireframe.color[]) == Makie.to_color(:red)
        @test wireframe.linewidth[] == 2
        Makie.update!(plot; spherevisible=false)
        @test !surface.visible[] && !wireframe.visible[]
        @test arrow.visible[]
        @test_throws "sphereresolution must be at least" blochsphereplot(up; sphereresolution=(2, 1))
        @test_throws "blochsphereplot requires a two-level state" blochsphereplot(spinup(SpinBasis(1)))
    end

    @testset "Mixed states shorten arrows without thinning them" begin
        b = SpinBasis(1//2)
        up, down = dm(spinup(b)), dm(spindown(b))
        state = Observable(up)
        fig, ax, plot = blochsphereplot(state)
        arrow = plot.plots[end]
        shaft, tip = arrow.plots[2:3]
        shaftsize, tipsize = only(shaft.markersize[]), only(tip.markersize[])
        fulllength = norm(only(arrow.endpoints[]) - only(arrow.startpoints[]))
        for r in (0.4, 0.04, 0.001, 0.0, 1.0)
            state[] = (1+r)/2 * up + (1-r)/2 * down
            if iszero(r)
                @test isempty(shaft.markersize[]) && isempty(tip.markersize[])
                @test !isempty(Makie.colorbuffer(fig))
            else
                @test only(shaft.markersize[])[1:2] ≈ shaftsize[1:2]
                @test only(tip.markersize[])[1:2] ≈ tipsize[1:2]
                @test norm(only(arrow.endpoints[]) - only(arrow.startpoints[])) ≈ r * fulllength rtol=1e-6
            end
        end
        Makie.update!(plot; shaftradius=0.02, tipradius=0.07)
        @test only(shaft.markersize[])[1:2] ≈ 2shaftsize[1:2]
        @test only(tip.markersize[])[1:2] ≈ 2tipsize[1:2]
    end

    @testset "Fock probabilities respect the basis offset" begin
        b = FockBasis(8, 3)
        state = normalize(fockstate(b, 4) + im*fockstate(b, 7))
        fig, ax, pure = fockdistributionplot(state)
        mixed = fockdistributionplot!(ax, dm(state))
        @test pure.points[] == mixed.points[]
        @test first.(pure.points[]) == 3:8
        @test sum(last, pure.points[]) ≈ norm(state)^2
        @test sum(p -> p[1]*p[2], pure.points[]) ≈ real(expect(number(b), state))

        input = Observable(state)
        fig, ax, plot = fockdistributionplot(input)
        input[] = fockstate(FockBasis(12, 5), 9)
        @test length(plot.plots[1][1][]) == 8
        @test sum(p -> p[1]*p[2], plot.points[]) ≈ 9
        @test_throws "fockdistributionplot requires a FockBasis state" fockdistributionplot(spinup(SpinBasis(1//2)))
    end

    @testset "Wavefunction density and components" begin
        for b in (PositionBasis(-8, 8, 128), MomentumBasis(-8, 8, 128))
            state = gaussianstate(b, 0, 1, 1)
            fig, ax, density = wavefunctionplot(state)
            re = wavefunctionplot!(ax, state; component=real)
            implot = wavefunctionplot!(ax, state; component=imag)
            @test first.(density.points[]) ≈ samplepoints(b)
            @test sum(last, density.points[]) * spacing(b) ≈ norm(state)^2
            @test last.(density.points[]) ≈ last.(re.points[]).^2 + last.(implot.points[]).^2
            @test_throws "wavefunctionplot requires a Ket in a PositionBasis or MomentumBasis" wavefunctionplot(dm(state))
        end
        input = Observable(gaussianstate(PositionBasis(-8, 8, 64), 0, 0, 1))
        fig, ax, plot = wavefunctionplot(input)
        input[] = gaussianstate(PositionBasis(-8, 8, 96), 1, 2, 1)
        Makie.update!(plot; component=real)
        @test length(plot.plots[1][1][]) == 96
        @test sum(abs2, last.(plot.points[])) * spacing(basis(input[])) <= norm(input[])^2
        @test_throws "wavefunctionplot requires a Ket in a PositionBasis or MomentumBasis" wavefunctionplot(fockstate(FockBasis(5), 0))
    end

    @testset "Wigner grid, parity, and updates" begin
        b = FockBasis(10)
        x, p = range(-4, 4; length=41), range(-3, 3; length=31)
        state = Observable(fockstate(b, 1))
        fig, ax, plot = wignerplot(state, x, p)
        heatmap = only(plot.plots)
        colorbar = Colorbar(fig[1, 2], plot)
        @test Makie.to_colormap(heatmap.colormap[]) == Makie.to_colormap(:RdBu)
        @test size(plot.values[]) == (41, 31)
        @test plot.values[][21, 16] ≈ -1/pi
        @test heatmap.colorrange[] ≈ [-1/pi, 1/pi] atol=1e-7
        state[] = fockstate(b, 0)
        @test plot.values[][21, 16] ≈ 1/pi
        @test heatmap.colorrange[] ≈ [-1/pi, 1/pi] atol=1e-7
        state[] = 0.5fockstate(b, 0)
        @test heatmap.colorrange[] ≈ [-0.25/pi, 0.25/pi] atol=1e-7
        @test collect(colorbar.limits[]) ≈ [-0.25/pi, 0.25/pi] atol=1e-7
        state[] = 0fockstate(b, 0)
        @test heatmap.colorrange[] == [-1, 1]
        Makie.update!(plot; colorrange=(-0.2, 0.5))
        state[] = fockstate(b, 1)
        @test heatmap.colorrange[] ≈ [-0.2, 0.5] atol=1e-7
        Makie.update!(plot; colorrange=Makie.automatic)
        @test heatmap.colorrange[] ≈ [-1/pi, 1/pi] atol=1e-7
        Makie.update!(plot; arg2=range(-4, 4; length=51), arg3=range(-3, 3; length=21))
        @test size(plot.plots[1][3][]) == (51, 21)
        @test_throws "wignerplot requires a FockBasis state" wignerplot(spinup(SpinBasis(1//2)), x, p)
    end

    @testset "Native styles, cycles, legends, and rendering" begin
        with_theme(Theme(palette=(color=[:red, :blue], patchcolor=[:orange, :purple]),
                BlochSpherePlot=(spherecolor=(:gray, 0.25), wireframecolor=:blue,
                    wireframewidth=2, sphereresolution=(16, 8), color=:green, cycle=[]),
                FockDistributionPlot=(gap=0.4,),
                WaveFunctionPlot=(linewidth=4,),
                WignerPlot=(colormap=:viridis, colorrange=(-0.2, 0.4)))) do
            fig = Figure()
            b = FockBasis(10)
            ax = Axis(fig[1, 1])
            bars = fockdistributionplot!(ax, fockstate(b, 2); label="Fock")
            bars2 = fockdistributionplot!(ax, fockstate(b, 3))
            @test Makie.to_color(bars.plots[1].color[]) == Makie.to_color(:orange)
            @test Makie.to_color(bars2.plots[1].color[]) == Makie.to_color(:purple)
            @test bars.plots[1].gap[] == 0.4
            axislegend(ax)
            ax = Axis(fig[1, 2])
            state = gaussianstate(PositionBasis(-6, 6, 64), 0, 1, 1)
            re = wavefunctionplot!(ax, state; component=real, label="real")
            implot = wavefunctionplot!(ax, state; component=imag, label="imag")
            @test Makie.to_color(re.plots[1].color[]) == Makie.to_color(:red)
            @test Makie.to_color(implot.plots[1].color[]) == Makie.to_color(:blue)
            @test re.plots[1].linewidth[] == 4
            axislegend(ax)
            ax = Axis3(fig[2, 1])
            bloch = blochsphereplot!(ax, dm(spinup(SpinBasis(1//2))))
            @test Makie.to_color(bloch.plots[1].color[]) == Makie.to_color((:gray, 0.25))
            @test count(p -> any(isnan, p), bloch.plots[2][1][]) == 23
            @test Makie.to_color(bloch.plots[2].color[]) == Makie.to_color(:blue)
            @test bloch.plots[2].linewidth[] == 2
            @test Makie.to_color(bloch.plots[end].color[]) == Makie.to_color(:green)
            for color in (:red, :blue)
                overlay = blochsphereplot!(ax, spinup(SpinBasis(1//2)); spherevisible=false, cycle=[:color])
                @test Makie.to_color(overlay.plots[end].color[]) == Makie.to_color(color)
            end
            ax = Axis(fig[2, 2])
            w = wignerplot!(ax, fockstate(b, 1), range(-3, 3; length=21), range(-3, 3; length=21))
            @test Makie.to_colormap(w.plots[1].colormap[]) == Makie.to_colormap(:viridis)
            @test w.plots[1].colorrange[] ≈ [-0.2, 0.4] atol=1e-7
            Colorbar(fig[2, 3], w)
            mktempdir() do dir
                path = joinpath(dir, "quantum-plots.png")
                save(path, fig)
                @test filesize(path) > 0
            end
        end
    end
end

@testitem "Plot extension loading" tags=[:plotting] begin
    # A separate process is needed: other test items may have already loaded Makie.
    script = raw"""
    using Test, QuantumOpticsBase
    @test Base.get_extension(QuantumOpticsBase, :QuantumOpticsBaseMakieExt) === nothing
    for plot in (blochsphereplot, blochsphereplot!,
            wignerplot, wignerplot!, fockdistributionplot, fockdistributionplot!,
            wavefunctionplot, wavefunctionplot!)
        for kwargs in ((;), (; color=:red))
            err = try
                plot(nothing; kwargs...)
            catch e
                e
            end
            @test occursin("import Makie", sprint(showerror, err))
        end
    end
    using CairoMakie
    @test Base.get_extension(QuantumOpticsBase, :QuantumOpticsBaseMakieExt) !== nothing
    state = fockstate(FockBasis(5), 0)
    err = try
        wignerplot(state, -1:1, -1:1)
    catch e
        e
    end
    @test occursin("import QuantumOptics", sprint(showerror, err))
    using QuantumOptics
    @test wignerplot(state, -1:1, -1:1).figure isa Figure
    """
    @test success(run(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $script`))
end
