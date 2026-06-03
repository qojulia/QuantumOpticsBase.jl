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