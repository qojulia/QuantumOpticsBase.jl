module QuantumOpticsBaseMakieExt
 
import QuantumOpticsBase
import QuantumOpticsBase: Ket
import Makie
using Makie: Figure, @recipe, Attributes, Axis3
using Makie: surface!, arrows3d!, lines!, text!, meshscatter!
using Makie: Point3f, Vec3f   # re-exported by Makie from GeometryBasics — no separate dep needed
 
export blochsphereplot, blochsphereplot!
 
# ─── Recipe definition ────────────────────────────────────────────────────────
 
@recipe(BlochSpherePlot, state) do scene
    Attributes(
        arrowcolor    = :red,
        spherecolor   = :lightblue,  # any Makie-compatible colour value
        spherealpha   = 0.15,        # low enough that back-facing labels show through
        showwireframe = true,        # equator + two meridian great circles
        showaxes      = true,        # bidirectional dashed x/y/z lines
        showlabels    = true,        # |0⟩ |1⟩ |+⟩ |-⟩ |+i⟩ |-i⟩ pole labels
        labelsize     = 18,
        shaftradius   = 0.018,
        tipradius     = 0.050,
        tiplength     = 0.10,
    )
end
 
# ─── Main recipe ──────────────────────────────────────────────────────────────
 
function Makie.plot!(p::BlochSpherePlot)
    state_obs = p[1]   # Observable{Ket}
 
    # ── Bloch vector from |ψ⟩ = α|0⟩ + β|1⟩ ─────────────────────────────────
    #   x = 2 Re(ᾱβ)   y = 2 Im(ᾱβ)   z = |α|² – |β|²
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
 
    # ── Smooth sphere surface ─────────────────────────────────────────────────
    # High resolution (200×200) makes individual UV cells too fine to see in
    # CairoMakie's flat-polygon renderer, avoiding the visible mesh-grid artifact.
    # We combine spherecolor + spherealpha into a single RGBAf at plot time.
    let npts = 200
        θ = LinRange(0f0, 2f0π, npts)
        φ = LinRange(0f0, Float32(π), npts)
        xs = Float32[cos(t) * sin(q) for t in θ, q in φ]
        ys = Float32[sin(t) * sin(q) for t in θ, q in φ]
        zs = Float32[cos(q)          for _  in θ, q in φ]
        c  = Makie.to_color(p[:spherecolor][])
        α  = Float32(p[:spherealpha][])
        rgba = Makie.RGBAf(Makie.red(c), Makie.green(c), Makie.blue(c), α)
        surface!(p, xs, ys, zs;
            color        = fill(rgba, npts, npts),
            transparency = true,
            # shading is intentionally omitted — FastShading was deprecated as a
            # plot attribute in Makie 0.24; the default scene shading is sufficient
        )
    end
 
    # ── Wireframe great circles ───────────────────────────────────────────────
    if p[:showwireframe][]
        ncirc = 120
        θc = LinRange(0f0, 2f0π, ncirc)
        for pts in (
            [Point3f( cos(t),  sin(t), 0f0) for t in θc],   # equator  (xy)
            [Point3f( cos(t), 0f0, sin(t)) for t in θc],    # meridian (xz)
            [Point3f(0f0, cos(t), sin(t)) for t in θc],     # meridian (yz)
        )
            lines!(p, pts; color = (:black, 0.70), linewidth = 1.2)
        end
    end
 
    # ── Bidirectional dashed axis lines ──────────────────────────────────────
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
 
    # ── State-vector arrow (reactive to state changes) ────────────────────────
    arrows3d!(p,
        [Point3f(0, 0, 0)],
        Makie.@lift([$blochvec]);
        shaftradius = p[:shaftradius],
        tipradius   = p[:tipradius],
        tiplength   = p[:tiplength],
        color       = p[:arrowcolor],
    )
 
    # ── Dot at the state point on the sphere surface ──────────────────────────
    # meshscatter! places a true 3D sphere marker in scene space, so it sits
    # correctly on the surface regardless of camera angle — unlike scatter!
    # which uses flat 2D screen-space markers that get occluded by the arrow tip.
    meshscatter!(p,
        Makie.@lift([Point3f($blochvec)]);
        color      = p[:arrowcolor],
        markersize = 0.06,
    )
 
    # ── Pole labels ───────────────────────────────────────────────────────────
    if p[:showlabels][]
        ls  = p[:labelsize][]
        off = 1.40f0   # outside the sphere surface; lim=1.6 gives enough room
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
 
# ─── Convenience constructor ──────────────────────────────────────────────────
 
function QuantumOpticsBase.blochsphere(state::Ket; kwargs...)
    fig = Figure(size = (700, 700))
    ax  = Axis3(fig[1, 1];
        aspect   = :data,
        viewmode = :fit,
        # Hide Makie's own ticks and axis labels — we draw our own pole labels
        xticksvisible      = false,
        yticksvisible      = false,
        zticksvisible      = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        zticklabelsvisible = false,
        xlabelvisible      = false,
        ylabelvisible      = false,
        zlabelvisible      = false,
        # Hide the bounding-box spines, interior grid planes, and background panels
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
    lim = Float32(get(kwargs, :limits, 1.6))
    Makie.limits!(ax, -lim, lim, -lim, lim, -lim, lim)
    plt = blochsphereplot!(ax, state; kwargs...)
    return fig, ax, plt
end
 
end # module