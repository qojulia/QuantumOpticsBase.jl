module QuantumOpticsBaseMakieExt
import QuantumOpticsBase
import QuantumOpticsBase: Ket
import Makie
import Makie: convert_arguments
using Makie: Figure, @recipe, Attributes, Axis3, mesh!, arrows3d!, hidexdecorations!, hideydecorations!, hidezdecorations!
using GeometryBasics: Point3f, Vec3f, Sphere

@recipe(BlochSpherePlot, state) do scene
    Attributes(
        arrowcolor  = :red,
        spherealpha = 0.30,
        showaxes    = true,
        title       = "Bloch Sphere",
        shaftradius = 0.018,
        tipradius   = 0.050,
        tiplength   = 0.10,
        lim         = 1.2,
    )
end

function convert_arguments(::Type{<:Makie.Plot{blochsphereplot}}, state::Ket)
    return (state,)
end


# Walk up the parent chain until we find an Axis3 (or give up).
function _parent_axis3(x)
    cur = x
    for _ in 1:10
        cur isa Axis3 && return cur
        cur = Makie.parent(cur)
        cur === nothing && break
    end
    return nothing
end

function Makie.plot!(p::BlochSpherePlot)
    state_obs = p[1]  # Observable{Ket}

    origin = Point3f(0, 0, 0)

    # Bloch vector as an Observable (reactive)
    blochvec = Makie.@lift begin
        s = $state_obs
        length(s.data) == 2 || error("BlochSphere only supports spin-1/2 states (2 amplitudes)")
        α, β = s.data
        x = 2 * real(conj(α) * β)
        y = 2 * imag(conj(α) * β)
        z = abs2(α) - abs2(β)
        Vec3f(x, y, z)
    end

    # Sphere
    mesh!(p, Sphere(origin, 1f0);
        color = :white,
        alpha = p[:spherealpha],
        transparency = true,
        shading = true
    )

    # Axes + state vector arrows (dirs/colors reactive where needed)
    dirs = Makie.@lift Vec3f[Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1), $blochvec]
    tails = fill(origin, 4)

    colors = Makie.@lift Any[:black, :black, :black, $(p[:arrowcolor])]

    arrows3d!(p, tails, dirs;
        shaftradius = p[:shaftradius],
        tipradius   = p[:tipradius],
        tiplength   = p[:tiplength],
        color       = colors
    )

    ax = _parent_axis3(p)
    if ax !== nothing 
        ax.aspect = (1,1,1)
        lim = p[:lim][]
        Makie.limits!(ax, -lim, lim, -lim, lim, -lim, lim)
        ax.title = p[:title][]

        if !p[:showaxes][]
            hidexdecorations!(ax)
            hideydecorations!(ax)
            hidezdecorations!(ax)
        end
    end

    return p
end
function QuantumOpticsBase.blochsphere(state::Ket; kwargs...)
    fig = Figure(size = (700, 700))
    ax  = Axis3(fig[1,1];
        aspect   = :data,
        viewmode = :fit
    )
    blochsphereplot!(ax, state; kwargs...)
    return fig
end

end
