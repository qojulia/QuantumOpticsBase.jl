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
function wignerplot_axis end
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
function blochsphereplot_axis end
