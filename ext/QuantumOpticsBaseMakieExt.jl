module QuantumOpticsBaseMakieExt

import QuantumOpticsBase: blochsphere, Ket
using Makie
using GeometryBasics
using LinearAlgebra: normalize

# Configure axis visibility safely
function configure_axis!(ax, showaxes::Bool)
    if !showaxes
        hidexdecorations!(ax)
        hideydecorations!(ax)
        hidezdecorations!(ax)
    end
end

function blochsphere(state::Ket; arrowcolor=:red, spherealpha=0.35, showaxes=true)
    length(state.data) == 2 || error("Bloch sphere only supports spin-1/2 states")

    # Compute Bloch vector
    α, β = state.data
    x = 2 * real(conj(α) * β)
    y = 2 * imag(conj(α) * β)
    z = abs2(α) - abs2(β)
    blochvec = Vec3f(x, y, z)

    origin = Point3f(0,0,0)

    # Axes + state vector
    dirs = [
        Vec3f(1,0,0),
        Vec3f(0,1,0),
        Vec3f(0,0,1),
        blochvec
    ]
    tips = [Point3f(origin .+ d) for d in dirs]

    # Figure & axis
    f = Figure()
    ax = Axis3(f[1,1]; title="Bloch Sphere", aspect=:data)
    configure_axis!(ax, showaxes)

    # Sphere
    mesh!(ax, Sphere(origin, 1f0); color=:white, alpha=spherealpha, transparency=true)

    # Draw arrows
    for (tail, tip) in zip(fill(origin, length(dirs)), tips)
        arrows3d!(ax, [tail], [tip];
                  shaftradius=0.02,
                  tipradius=0.06,
                  tiplength=0.1,
                  color = tip == tips[end] ? arrowcolor : :black)
    end

    limits!(ax, -1.2,1.2,-1.2,1.2,-1.2,1.2)

    return f
end

end # module
