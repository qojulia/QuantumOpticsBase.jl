# Forward plot constructors into the extension, including calls with keywords.
@declare_struct_is_in_extension QuantumOpticsBase blochsphereplot :QuantumOpticsBaseMakieExt (:Makie,) """
    blochsphereplot(state; kwargs...)

Draw a two-level `Ket` or density `Operator` as a Bloch vector inside a translucent
gray sphere with a wireframe. Mixed states shorten the arrow without changing its
shaft or tip radius; the maximally mixed state has no arrow. Arrow colors use
Makie's palette. The first basis state is +z, the second is -z. States are used as given;
use a normalized ket or a density operator with unit trace.

Load a Makie backend, for example `using CairoMakie`, before plotting. Returns
Makie's `FigureAxisPlot` with an `Axis3`. Accepts Makie `Arrows3D` attributes and
the themeable attributes `spherecolor`, `wireframecolor`, `wireframewidth`,
`sphereresolution=(12, 6)` (azimuthal and polar subdivisions), and `spherevisible`.
The arrow defaults to `markerscale=1`, `minshaftlength=0`, `shaftradius=0.01`,
`tipradius=0.035`, and `tiplength=0.1`. The tip length is capped at half the arrow
length so short vectors keep their width and a visible shaft.
Use `(color, alpha)` tuples for transparency. Configure the axis and figure with
Makie's `axis` and `figure` keywords.
"""

@declare_struct_is_in_extension QuantumOpticsBase blochsphereplot! :QuantumOpticsBaseMakieExt (:Makie,) """
    blochsphereplot!([ax,] state; kwargs...)

Draw [`blochsphereplot`](@ref) on an existing Makie axis and return the plot.
"""

@declare_struct_is_in_extension QuantumOpticsBase wignerplot :QuantumOpticsBaseMakieExt (:Makie,) """
    wignerplot(state, x, p; kwargs...)

Plot the Wigner function of a Fock-basis ket or density operator on coordinate
vectors `x` and `p`. Requires `using QuantumOptics` for `wigner` and a
Makie backend such as CairoMakie for plotting. Accepts Makie `Heatmap` attributes,
including `colormap` and `colorrange`. The default colormap is `:RdBu`.
Automatic color limits are symmetric about zero and cover the largest absolute
Wigner value, using `(-1, 1)` for all-zero data. An explicit `colorrange` overrides
these limits. Add a Makie `Colorbar` when needed.
"""

@declare_struct_is_in_extension QuantumOpticsBase wignerplot! :QuantumOpticsBaseMakieExt (:Makie,) """
    wignerplot!([ax,] state, x, p; kwargs...)

Draw [`wignerplot`](@ref) on an existing Makie axis and return the plot.
"""

@declare_struct_is_in_extension QuantumOpticsBase fockdistributionplot :QuantumOpticsBaseMakieExt (:Makie,) """
    fockdistributionplot(state; kwargs...)

Plot Fock occupation probabilities against the basis's number range, including
its offset. Uses squared amplitudes for a `Ket` and the real diagonal for a
density `Operator`, without renormalizing. Load a Makie backend first.
Accepts Makie `BarPlot` attributes and uses its default color cycle.
"""

@declare_struct_is_in_extension QuantumOpticsBase fockdistributionplot! :QuantumOpticsBaseMakieExt (:Makie,) """
    fockdistributionplot!([ax,] state; kwargs...)

Draw [`fockdistributionplot`](@ref) on an existing Makie axis and return the plot.
"""

@declare_struct_is_in_extension QuantumOpticsBase wavefunctionplot :QuantumOpticsBaseMakieExt (:Makie,) """
    wavefunctionplot(state; component=abs2, kwargs...)

Plot a `Ket` in a `PositionBasis` or `MomentumBasis` at its sample points.
Amplitudes are divided by `sqrt(spacing(basis(state)))` before applying the
real-valued `component` function. Thus `abs2` gives probability density;
`real`, `imag`, and `abs` give amplitude components or magnitude.

Load a Makie backend first. Accepts Makie `Lines` attributes and its default
color cycle. Set axis labels and legends to describe the selected component.
"""

@declare_struct_is_in_extension QuantumOpticsBase wavefunctionplot! :QuantumOpticsBaseMakieExt (:Makie,) """
    wavefunctionplot!([ax,] state; kwargs...)

Draw [`wavefunctionplot`](@ref) on an existing Makie axis and return the plot.
"""

const WEAKDEP_METHOD_ERROR_HINTS = WeakDepCache()
register_method_error_hint(WEAKDEP_METHOD_ERROR_HINTS, wigner, (:QuantumOptics,))
