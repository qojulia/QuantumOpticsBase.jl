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

@declare_method_is_in_extension WEAKDEP_METHOD_ERROR_HINTS wavefunctionplot (:Makie,) """
    wavefunctionplot(state::Ket; kwargs...)

Visualize the wave function of a particle state as a line plot against position
or momentum.

The `part` attribute selects what is drawn: `:abs2` (default) for the
probability density, or `:abs`, `:real`, `:imag` for the modulus, real part or
imaginary part of the amplitude.

Requires a Makie backend be already imported.
"""
@declare_method_is_in_extension WEAKDEP_METHOD_ERROR_HINTS wavefunctionplot! (:Makie,) """
    wavefunctionplot!(ax, state::Ket; kwargs...)

In-place version of [`wavefunctionplot`](@ref). Plots onto an existing Makie axis,
which allows several parts to be overlaid on the same axis.

Requires a Makie backend be already imported.
"""
@declare_method_is_in_extension WEAKDEP_METHOD_ERROR_HINTS wavefunctionplot_axis (:Makie,) """
    wavefunctionplot_axis([ax,] state::Ket; kwargs...) -> (Figure, Axis, Plot)

Visualize the wave function of a particle state, creating a new Figure and Axis
or plotting onto an existing one. Axis labels follow the basis: `x` for a
`PositionBasis`, `p` for a `MomentumBasis`.

Requires a Makie backend be already imported.
"""
