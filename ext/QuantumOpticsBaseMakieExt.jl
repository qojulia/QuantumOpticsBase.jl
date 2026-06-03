module QuantumOpticsBaseMakieExt

import QuantumOpticsBase
import QuantumOpticsBase: Ket, wigner,
    blochsphereplot, blochsphereplot!, blochsphereplot_axis,
    wignerplot, wignerplot!, wignerplot_axis
import Makie
using Makie: Figure, @recipe, Attributes, Axis, Axis3, Colorbar, DataAspect
using Makie: surface!, arrows3d!, lines!, text!, meshscatter!, heatmap!
using Makie: Point3f, Vec3f

# ═══════════════════════════════════════════════════════════════════════════════
# Bloch sphere recipe
# ═══════════════════════════════════════════════════════════════════════════════

@recipe(BlochSpherePlot, state) do scene
    Attributes(
        arrowcolor    = :red,
        spherecolor   = :lightblue,
        spherealpha   = 0.15,
        showwireframe = true,
        showaxes      = true,
        showlabels    = true,
        labelsize     = 18,
        shaftradius   = 0.018,
        tipradius     = 0.050,
        tiplength     = 0.10,
    )
end

function Makie.plot!(p::BlochSpherePlot)
    state_obs = p[1]

    blochvec = Makie.@lift begin
        s = $state_obs
        length(s.data) == 2 ||
            error("BlochSphere requires a 2-level (spin-1/2) state")
        α, β = s.data
        Vec3f(
            Float32(2 * real(conj(α) * β)),
            Float32(2 * imag(conj(α) * β)),
            Float32(abs2(α) - abs2(β)),
        )
    end

    let npts = 200
        θ = LinRange(0f0, 2f0π, npts)
        φ = LinRange(0f0, Float32(π), npts)
        xs = Float32[cos(t) * sin(q) for t in θ, q in φ]
        ys = Float32[sin(t) * sin(q) for t in θ, q in φ]
        zs = Float32[cos(q)          for _  in θ, q in φ]
        c  = Makie.to_color(p[:spherecolor][])
        α  = Float32(p[:spherealpha][])
        rgba = Makie.RGBAf(c.r, c.g, c.b, α)
        surface!(p, xs, ys, zs;
            color        = fill(rgba, npts, npts),
            transparency = true,
        )
    end

    if p[:showwireframe][]
        ncirc = 120
        θc = LinRange(0f0, 2f0π, ncirc)
        for pts in (
            [Point3f( cos(t),  sin(t), 0f0) for t in θc],
            [Point3f( cos(t), 0f0, sin(t)) for t in θc],
            [Point3f(0f0, cos(t), sin(t)) for t in θc],
        )
            lines!(p, pts; color = (:black, 0.70), linewidth = 1.2)
        end
    end

    if p[:showaxes][]
        r = 1.18f0
        for (a, b) in (
            (Point3f(-r, 0, 0), Point3f(r, 0, 0)),
            (Point3f(0, -r, 0), Point3f(0, r, 0)),
            (Point3f(0, 0, -r), Point3f(0, 0, r)),
        )
            lines!(p, [a, b]; color = :black, linewidth = 1, linestyle = :dash)
        end
    end

    arrows3d!(p,
        [Point3f(0, 0, 0)],
        Makie.@lift([$blochvec]);
        shaftradius = p[:shaftradius],
        tipradius   = p[:tipradius],
        tiplength   = p[:tiplength],
        color       = p[:arrowcolor],
    )

    meshscatter!(p,
        Makie.@lift([Point3f($blochvec)]);
        color      = p[:arrowcolor],
        markersize = 0.06,
    )

    if p[:showlabels][]
        ls  = p[:labelsize][]
        off = 1.40f0
        for (pos, lbl, align) in (
            (Point3f( 0f0,   0f0,  off), "|0⟩",  (:center, :bottom)),
            (Point3f( 0f0,   0f0, -off), "|1⟩",  (:center, :top   )),
            (Point3f( off,   0f0,  0f0), "|+⟩",  (:left,   :center)),
            (Point3f(-off,   0f0,  0f0), "|-⟩",  (:right,  :center)),
            (Point3f( 0f0,   off,  0f0), "|+i⟩", (:left,   :center)),
            (Point3f( 0f0,  -off,  0f0), "|-i⟩", (:right,  :center)),
        )
            text!(p, pos; text = lbl, fontsize = ls, align = align)
        end
    end

    return p
end

function QuantumOpticsBase.blochsphereplot_axis(ax::Makie.AbstractAxis, state; limits=1.6, kwargs...)
    ax.perspectiveness = 0f0
    lim = Float32(limits)
    Makie.limits!(ax, -lim, lim, -lim, lim, -lim, lim)
    blochsphereplot!(ax, state; kwargs...)
end

function QuantumOpticsBase.blochsphereplot_axis(state; limits=1.6, kwargs...)
    fig = Figure(size = (700, 700))
    ax  = Axis3(fig[1, 1];
        aspect   = :data,
        viewmode = :fit,
        perspectiveness = 0f0,
        xticksvisible      = false,
        yticksvisible      = false,
        zticksvisible      = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        zticklabelsvisible = false,
        xlabelvisible      = false,
        ylabelvisible      = false,
        zlabelvisible      = false,
        xspinesvisible     = false,
        yspinesvisible     = false,
        zspinesvisible     = false,
        xgridvisible       = false,
        ygridvisible       = false,
        zgridvisible       = false,
        xypanelvisible     = false,
        xzpanelvisible     = false,
        yzpanelvisible     = false,
    )
    plt = QuantumOpticsBase.blochsphereplot_axis(ax, state; limits, kwargs...)
    return fig, ax, plt
end

# ═══════════════════════════════════════════════════════════════════════════════
# Wigner plot recipe
# ═══════════════════════════════════════════════════════════════════════════════

@recipe(WignerPlot, state) do scene
    Attributes(
        xrange   = (-5.0, 5.0),
        prange   = (-5.0, 5.0),
        npoints  = 100,
        colormap = :RdBu,
    )
end

function Makie.plot!(p::WignerPlot)
    state_obs = p[1]

    grid = Makie.@lift begin
        s            = $state_obs
        xmin, xmax   = p[:xrange][]
        pmin, pmax   = p[:prange][]
        n            = p[:npoints][]
        xvec         = collect(LinRange(Float64(xmin), Float64(xmax), n))
        pvec         = collect(LinRange(Float64(pmin), Float64(pmax), n))
        W            = wigner(s, xvec, pvec)
        mx           = max(abs(minimum(W)), abs(maximum(W)))
        (xvec, pvec, W, mx)
    end

    xs   = Makie.@lift $grid[1]
    pvs  = Makie.@lift $grid[2]
    Ws   = Makie.@lift $grid[3]
    clim = Makie.@lift (-$grid[4], $grid[4])

    heatmap!(p, xs, pvs, Ws;
        colormap   = p[:colormap],
        colorrange = clim,
    )

    return p
end

function QuantumOpticsBase.wignerplot_axis(ax, state; kwargs...)
    wignerplot!(ax, state; kwargs...)
end

function QuantumOpticsBase.wignerplot_axis(state; kwargs...)
    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1];
        xlabel          = "x",
        ylabel          = "p",
        aspect          = DataAspect(),
        xgridvisible    = false,
        ygridvisible    = false,
        backgroundcolor = :white,
    )
    plt = wignerplot_axis(ax, state; kwargs...)

    Colorbar(fig[1, 2], plt;
        label = "W(x,p)",
        width = 15,
    )

    return fig, ax, plt
end

end # module
