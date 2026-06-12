const _MAKIE_HINT = "requires a Makie backend (e.g. `using CairoMakie`). Load one before calling this function."

"""
    wignerplot(state; kwargs...)

Visualize the Wigner quasi-probability distribution of a quantum state as a heatmap.

Requires a Makie backend be already imported.
"""
function wignerplot end

"""
    wignerplot!(ax, state; kwargs...)

In-place version of [`wignerplot`](@ref). Plots onto an existing Makie axis.

Requires a Makie backend be already imported.
"""
function wignerplot! end

"""
    wignerplot_axis([ax,] state; kwargs...) -> (Figure, Axis, Plot)

Visualize the Wigner quasi-probability distribution of a quantum state,
creating a new Figure and Axis or plotting onto an existing one.

Requires a Makie backend be already imported.
"""
function wignerplot_axis(args...; kwargs...)
    error("wignerplot_axis " * _MAKIE_HINT)
end

"""
    blochsphereplot(state::Ket; kwargs...)

Visualize a pure qubit state as an arrow on a Bloch sphere.

Requires a Makie backend be already imported.
"""
function blochsphereplot end

"""
    blochsphereplot!(ax, state::Ket; kwargs...)

In-place version of [`blochsphereplot`](@ref). Plots onto an existing Makie axis.

Requires a Makie backend be already imported.
"""
function blochsphereplot! end

"""
    blochsphereplot_axis([ax,] state::Ket; kwargs...) -> (Figure, Axis3, Plot)

Visualize a pure qubit state on a Bloch sphere, creating a new Figure and Axis3
or plotting onto an existing one.

Requires a Makie backend be already imported.
"""
function blochsphereplot_axis(args...; kwargs...)
    error("blochsphereplot_axis " * _MAKIE_HINT)
end