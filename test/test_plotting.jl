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
            @test only(plot.directions[]) ≈ real.([expect(op, state) for op in (sigmax(b), sigmay(b), sigmaz(b))]) atol=1e-7
        end
        state = Observable(up)
        fig, ax, plot = blochsphereplot(state)
        @test ax isa Axis3
        state[] = down
        @test only(plot.directions[]) ≈ [0, 0, -1]
        Makie.update!(plot; spherevisible=false)
        @test !plot.plots[1].visible[]
        @test_throws "blochsphereplot requires a two-level state" blochsphereplot(spinup(SpinBasis(1)))
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
        @test size(plot.values[]) == (41, 31)
        @test plot.values[][21, 16] ≈ -1/pi
        state[] = fockstate(b, 0)
        @test plot.values[][21, 16] ≈ 1/pi
        Makie.update!(plot; arg2=range(-4, 4; length=51), arg3=range(-3, 3; length=21))
        @test size(plot.plots[1][3][]) == (51, 21)
        Colorbar(fig[1, 2], plot)
        @test_throws "wignerplot requires a FockBasis state" wignerplot(spinup(SpinBasis(1//2)), x, p)
    end

    @testset "Native styles, cycles, legends, and rendering" begin
        with_theme(Theme(palette=(color=[:red, :blue], patchcolor=[:orange, :purple]),
                BlochSpherePlot=(spherecolor=:gray, color=:green),
                FockDistributionPlot=(gap=0.4,),
                WaveFunctionPlot=(linewidth=4,),
                WignerPlot=(colormap=:RdBu, colorrange=(-1/pi, 1/pi)))) do
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
            @test Makie.to_color(bloch.plots[1].color[]) == Makie.to_color(:gray)
            @test Makie.to_color(bloch.plots[end].color[]) == Makie.to_color(:green)
            ax = Axis(fig[2, 2])
            w = wignerplot!(ax, fockstate(b, 1), range(-3, 3; length=21), range(-3, 3; length=21))
            @test w.plots[1].colorrange[] == (-1/pi, 1/pi)
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
    for plot in (blochsphereplot, blochsphereplot!, blochsphereplot_axis,
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
    run(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $script`)
end
