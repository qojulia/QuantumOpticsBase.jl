module QuantumOpticsBaseMakieExt
import QuantumOpticsBase
import QuantumOpticsBase: Ket
import Makie
using Makie: Figure, @recipe, Attributes, Axis3
using Makie: surface!, arrows3d!, lines!, text!, meshscatter!
using Makie: Point3f, Vec3f   

export blochsphereplot, blochsphereplot!
 
@recipe(BlochSpherePlot, state) do scene
    Attributes(
        arrowcolor    = :red,
        spherecolor   = :lightblue,  
        spherealpha   = 0.15,        
        showwireframe = true,        
        showaxes      = true,        
        showlabels    = true,        
        labelsize     = 18,
        title         = "Bloch Sphere",
        shaftradius   = 0.018,
        tipradius     = 0.050,
        tiplength     = 0.10,
        lim           = 1.6,
    )
end
 
function Makie.convert_arguments(::Type{<:Makie.Plot{blochsphereplot}}, state::Ket)
    return (state,)
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
        rgba = Makie.RGBAf(Makie.red(c), Makie.green(c), Makie.blue(c), α)
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
            lines!(p, pts; color = (:black, 0.55), linewidth = 1.2)
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
 

 
function QuantumOpticsBase.blochsphere(state::Ket; kwargs...)
    fig = Figure(size = (700, 700))
    ax  = Axis3(fig[1, 1];
        aspect   = :data,
        viewmode = :fit,
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
    lim = Float32(get(kwargs, :lim, 1.6))
    Makie.limits!(ax, -lim, lim, -lim, lim, -lim, lim)
    blochsphereplot!(ax, state; kwargs...)
    return fig
end
 
end 