module QuantumOpticsBaseMakieExt

using QuantumOpticsBase: Ket, Operator, basis, dm, FockBasis,
    PositionBasis, MomentumBasis, samplepoints, spacing, wigner
using LinearAlgebra: diag
import Makie
using Makie: @recipe, Point2d, Point3f, Vec3f

@recipe BlochSpherePlot (state,) begin
    "Surface color; use a (color, alpha) tuple to set its opacity."
    spherecolor = (:gray, 0.15)
    "Color of the latitude and longitude wireframe."
    wireframecolor = (:gray, 0.6)
    "Width of the wireframe lines in screen units."
    wireframewidth = 1
    "Number of azimuthal and polar subdivisions of the sphere mesh."
    sphereresolution = (24, 12)
    "Whether to draw the sphere surface and wireframe."
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
    Makie.map!(plot, [:visible, :spherevisible], :sphere_visible) do visible, spherevisible
        visible && spherevisible
    end
    Makie.map!(plot, :sphereresolution, [:sphere_x, :sphere_y, :sphere_z]) do resolution
        nθ, nφ = resolution
        nθ >= 3 && nφ >= 2 || throw(ArgumentError("sphereresolution must be at least (3, 2)"))
        θ = range(0, 2π; length=nθ+1)
        φ = range(0, π; length=nφ+1)
        ([cos(t)*sin(q) for t in θ, q in φ],
         [sin(t)*sin(q) for t in θ, q in φ],
         [cos(q) for t in θ, q in φ])
    end
    Makie.map!(plot, [:sphere_x, :spherecolor], :sphere_colors) do x, color
        fill(Makie.to_color(color), size(x))
    end
    Makie.surface!(plot, plot.sphere_x, plot.sphere_y, plot.sphere_z;
        color=plot.sphere_colors, alpha=plot.alpha, transparency=true, visible=plot.sphere_visible)
    Makie.wireframe!(plot, plot.sphere_x, plot.sphere_y, plot.sphere_z;
        color=plot.wireframecolor, linewidth=plot.wireframewidth, alpha=plot.alpha, visible=plot.sphere_visible)
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
    "Diverging colormap for negative and positive Wigner values."
    colormap = :RdBu
    Makie.filtered_attributes(Makie.Heatmap; exclude=(:colormap,))...
end

function Makie.plot!(plot::WignerPlot)
    Makie.map!(plot, [:state, :x, :p], :values) do state, x, p
        basis(state) isa FockBasis || throw(ArgumentError("wignerplot requires a FockBasis state"))
        wigner(state, x, p)
    end
    Makie.map!(plot, [:values, :colorrange], :wigner_colorrange) do values, colorrange
        colorrange === Makie.automatic || return colorrange
        limit = maximum(abs, values)
        iszero(limit) && (limit = one(limit))
        (-limit, limit)
    end
    Makie.heatmap!(plot, Makie.shared_attributes(plot, Makie.Heatmap; drop=[:colorrange]),
        plot.x, plot.p, plot.values; colorrange=plot.wigner_colorrange)
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
