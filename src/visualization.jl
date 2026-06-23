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
function blochsphereplot_axis end

"""
    fockdistributionplot(state; kwargs...)
 
Visualize the Fock-state (number-state) distribution of a quantum state as a
bar chart of occupation probabilities P(n).
 
For a `Ket` the probabilities are `|⟨n|ψ⟩|²`; for a density operator they are the
real diagonal entries `⟨n|ρ|n⟩`.
 
Requires a Makie backend be already imported.
"""
function fockdistributionplot end
 
"""
    fockdistributionplot!(ax, state; kwargs...)
 
In-place version of [`fockdistributionplot`](@ref). Plots onto an existing Makie axis.
 
Requires a Makie backend be already imported.
"""
function fockdistributionplot! end
 
"""
    fockdistributionplot_axis([ax,] state; kwargs...) -> (Figure, Axis, Plot)
 
Visualize the Fock-state distribution of a quantum state, creating a new Figure
and Axis or plotting onto an existing one.
 
Requires a Makie backend be already imported.
"""
function fockdistributionplot_axis end