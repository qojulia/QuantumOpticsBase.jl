module QuantumOpticsBaseMakieExt

using QuantumOpticsBase: Ket, Operator, basis, dm, FockBasis,
    PositionBasis, MomentumBasis, samplepoints, spacing, wigner
using LinearAlgebra: diag
import Makie
using Makie: @recipe, Point2d, Point3f, Vec3f

@recipe BlochSpherePlot (state,) begin
    "Color of the three great circles."
    spherecolor = @inherit linecolor
    "Whether to draw the three great circles."
    spherevisible = true
    Makie.documented_attributes(Makie.Arrows3D)...
end

function blochvector(state::Union{Ket,Operator})
    length(basis(state)) == 2 || throw(ArgumentError("blochsphereplot requires a two-level state"))
    ρ = (state isa Ket ? dm(state) : state).data
    return Vec3f(real(ρ[1, 2] + ρ[2, 1]), real(im * (ρ[1, 2] - ρ[2, 1])), real(ρ[1, 1] - ρ[2, 2]))
end

function Makie.plot!(plot::BlochSpherePlot)
    Makie.map!(plot, :state, :directions) do state
        [blochvector(state)]
    end
    Makie.map!(plot, [:visible, :spherevisible], :circlesvisible) do visible, spherevisible
        visible && spherevisible
    end
    θ = range(0, 2π; length=101)
    for points in (
        Point3f.(cos.(θ), sin.(θ), 0),
        Point3f.(cos.(θ), 0, sin.(θ)),
        Point3f.(0, cos.(θ), sin.(θ)),
    )
        Makie.lines!(plot, points; color=plot.spherecolor, visible=plot.circlesvisible)
    end
    Makie.arrows3d!(plot, Makie.shared_attributes(plot, Makie.Arrows3D), [Point3f(0)], plot.directions)
    return plot
end

Makie.preferred_axis_type(::BlochSpherePlot) = Makie.Axis3
Makie.preferred_axis_attributes(::Type{Makie.Axis3}, ::BlochSpherePlot) = (; aspect=:data)

function blochsphereplot_axis(args...; kwargs...)
    Base.depwarn("Use blochsphereplot or blochsphereplot! with Makie's axis keyword instead.", :blochsphereplot_axis)
    return blochsphereplot(args...; kwargs...)
end

function blochsphereplot_axis(ax::Makie.AbstractAxis, state; kwargs...)
    Base.depwarn("Use blochsphereplot!(ax, state) instead.", :blochsphereplot_axis)
    return blochsphereplot!(ax, state; kwargs...)
end

@recipe WignerPlot (state, x, p) begin
    Makie.documented_attributes(Makie.Heatmap)...
end

function Makie.plot!(plot::WignerPlot)
    Makie.map!(plot, [:state, :x, :p], :values) do state, x, p
        basis(state) isa FockBasis || throw(ArgumentError("wignerplot requires a FockBasis state"))
        wigner(state, x, p)
    end
    Makie.heatmap!(plot, Makie.shared_attributes(plot, Makie.Heatmap), plot.x, plot.p, plot.values)
    return plot
end

Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::WignerPlot) = (; aspect=Makie.DataAspect())

@recipe FockDistributionPlot (state,) begin
    Makie.documented_attributes(Makie.BarPlot)...
end

probabilities(state::Ket) = abs2.(state.data)
probabilities(state::Operator) = real.(diag(state.data))

function Makie.plot!(plot::FockDistributionPlot)
    Makie.map!(plot, :state, :points) do state
        b = basis(state)
        b isa FockBasis || throw(ArgumentError("fockdistributionplot requires a FockBasis state"))
        Point2d.(b.offset:b.N, probabilities(state))
    end
    Makie.barplot!(plot, Makie.shared_attributes(plot, Makie.BarPlot), plot.points)
    return plot
end

@recipe WaveFunctionPlot (state,) begin
    "Real-valued function applied to each amplitude: abs2, abs, real, or imag."
    component = abs2
    Makie.documented_attributes(Makie.Lines)...
end

function Makie.plot!(plot::WaveFunctionPlot)
    Makie.map!(plot, [:state, :component], :points) do state, component
        state isa Ket && basis(state) isa Union{PositionBasis,MomentumBasis} ||
            throw(ArgumentError("wavefunctionplot requires a Ket in a PositionBasis or MomentumBasis"))
        b = basis(state)
        Point2d.(samplepoints(b), component.(state.data ./ sqrt(spacing(b))))
    end
    Makie.lines!(plot, Makie.shared_attributes(plot, Makie.Lines), plot.points)
    return plot
end

end # module
