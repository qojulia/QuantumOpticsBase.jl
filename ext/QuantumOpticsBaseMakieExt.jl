module QuantumOpticsBaseMakieExt

using QuantumOpticsBase: Ket, Operator, basis, dm, FockBasis,
    PositionBasis, MomentumBasis, samplepoints, spacing, wigner
using LinearAlgebra: diag, norm
import Makie
using Makie: @recipe, Point2d, Point3f, Vec3f

@recipe BlochSpherePlot (state,) begin
    "Surface color; use a (color, alpha) tuple to set its opacity."
    spherecolor = (:gray, 0.03)
    "Color of the latitude and longitude wireframe."
    wireframecolor = (:gray, 0.35)
    "Width of the wireframe lines in screen units."
    wireframewidth = 1
    "Number of azimuthal and polar subdivisions of the wireframe."
    sphereresolution = (12, 6)
    "Whether to draw the sphere surface and wireframe."
    spherevisible = true
    "Cycle arrow colors using Makie's palette."
    cycle = [:color]
    "Arrow dimensions are in data units, independent of the state's purity."
    markerscale = 1
    "Allow short arrows to keep their shaft and tip radii."
    minshaftlength = 0
    "Radius of the arrow shaft."
    shaftradius = 0.01
    "Radius of the arrow tip."
    tipradius = 0.035
    "Length of the arrow tip, capped at half the arrow length for short vectors."
    tiplength = 0.1
    Makie.filtered_attributes(Makie.Arrows3D;
        exclude=(:markerscale, :minshaftlength, :shaftradius, :tipradius, :tiplength))...
end

function blochvector(state::Union{Ket,Operator})
    length(basis(state)) == 2 || throw(ArgumentError("blochsphereplot requires a two-level state"))
    ρ = (state isa Ket ? dm(state) : state).data
    return Vec3f(real(ρ[1, 2] + ρ[2, 1]), real(im * (ρ[1, 2] - ρ[2, 1])), real(ρ[1, 1] - ρ[2, 2]))
end

function Makie.plot!(plot::BlochSpherePlot)
    Makie.map!(plot, :state, :directions) do state
        v = blochvector(state)
        iszero(v) ? Vec3f[] : [v]
    end
    Makie.map!(plot, [:visible, :spherevisible], :sphere_visible) do visible, spherevisible
        visible && spherevisible
    end
    Makie.map!(plot, :sphereresolution, :wireframe_points) do resolution
        nθ, nφ = resolution
        nθ >= 3 && nφ >= 2 || throw(ArgumentError("sphereresolution must be at least (3, 2)"))
        points = Point3f[]
        for θ in range(0, 2π; length=nθ+1)[1:end-1]
            append!(points, [Point3f(cos(θ)*sin(φ), sin(θ)*sin(φ), cos(φ)) for φ in range(0, π; length=49)])
            push!(points, Point3f(NaN))
        end
        for φ in range(0, π; length=nφ+1)[2:end-1]
            append!(points, [Point3f(cos(θ)*sin(φ), sin(θ)*sin(φ), cos(φ)) for θ in range(0, 2π; length=97)])
            push!(points, Point3f(NaN))
        end
        points
    end
    Makie.map!(plot, [:directions, :model, :normalize, :lengthscale, :markerscale, :tiplength],
            :arrow_tiplength) do directions, model, normalize, lengthscale, markerscale, tiplength
        isempty(directions) && return tiplength
        v = only(directions)
        length = abs(lengthscale) * norm(model[1:3, 1:3] * (normalize ? v / norm(v) : v))
        scale = markerscale === Makie.automatic ? length : markerscale
        min(tiplength, length / (2scale))
    end
    sphere = Makie.normal_mesh(Makie.Tessellation(Makie.Sphere(Point3f(0), 1f0), 48))
    Makie.mesh!(plot, sphere; color=plot.spherecolor, alpha=plot.alpha,
        shading=Makie.NoShading, transparency=true, visible=plot.sphere_visible)
    Makie.lines!(plot, plot.wireframe_points; color=plot.wireframecolor,
        linewidth=plot.wireframewidth, alpha=plot.alpha, transparency=true, visible=plot.sphere_visible)
    Makie.arrows3d!(plot, Makie.shared_attributes(plot, Makie.Arrows3D; drop=[:tiplength]),
        [Point3f(0)], plot.directions; tiplength=plot.arrow_tiplength)
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
