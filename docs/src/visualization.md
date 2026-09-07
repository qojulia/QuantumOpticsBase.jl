# [Visualizations](@id visualizations)

Load a Makie backend to enable the plotting extension. These examples use
[CairoMakie](https://docs.makie.org/stable/explanations/backends/cairomakie.html).
Wigner plots also need `QuantumOptics`, which provides the Wigner function calculation
and reexports `QuantumOpticsBase`.

```julia
using QuantumOptics
using CairoMakie
```

```@setup visualization
using QuantumOptics
using CairoMakie
```

| Function | Input | Makie plot attributes |
|:--|:--|:--|
| [`blochsphereplot`](@ref) | Two-level ket or density operator | `Arrows3D`, plus sphere and wireframe attributes |
| [`fockdistributionplot`](@ref) | Fock-basis ket or density operator | `BarPlot` |
| [`wignerplot`](@ref) | Fock-basis ket or density operator, with two coordinate vectors | `Heatmap` |
| [`wavefunctionplot`](@ref) | Position- or momentum-basis ket | `Lines`, plus `component` |

Each function returns Makie's figure, axis, and plot objects. The corresponding `!`
function adds a plot to an existing axis. Use ordinary Makie attributes for axes,
labels, legends, colorbars, and plot styling. Inputs are not normalized automatically.

The earlier `blochsphereplot_axis` helper is deprecated. Use `blochsphereplot`
with `axis=(...)` instead. Use an ordinary `Axis3` for coordinate labels and
`color` for the arrow. The new recipes do not add `_axis` helpers.

## Bloch sphere

The arrow shows the expectation values of the Pauli matrices. A normalized pure
state reaches the sphere; a mixed state lies inside it. A translucent gray surface
and a latitude/longitude wireframe mark the unit sphere. `blochsphereplot` creates
an `Axis3` automatically.

The following attributes work as keywords or in a `Theme(BlochSpherePlot=(...))`:

| Attribute | Default | Effect |
|:--|:--|:--|
| `spherecolor` | `(:gray, 0.15)` | Surface color and opacity |
| `wireframecolor` | `(:gray, 0.6)` | Mesh line color and opacity |
| `wireframewidth` | `1` | Mesh line width in screen units |
| `sphereresolution` | `(24, 12)` | Azimuthal and polar subdivisions; increase for a denser mesh |
| `spherevisible` | `true` | Show or hide both the surface and its wireframe |

### Pure state

```@example visualization
b = SpinBasis(1//2)
psi = (spinup(b) + spindown(b)) / sqrt(2)
fig, ax, plot = blochsphereplot(psi;
    axis=(title="Pure spin along x", xlabel="⟨σx⟩", ylabel="⟨σy⟩", zlabel="⟨σz⟩"),
    figure=(size=(640, 480),))
save("bloch-pure.png", fig) #hide
nothing #hide
```

![Pure state on the Bloch sphere](bloch-pure.png)

### Mixed state

```@example visualization
b = SpinBasis(1//2)
rho = 0.7dm(spinup(b)) + 0.3dm(spindown(b))
fig = Figure(size=(640, 480))
ax = Axis3(fig[1, 1]; title="70% spin up, 30% spin down",
    xlabel="⟨σx⟩", ylabel="⟨σy⟩", zlabel="⟨σz⟩", aspect=:equal)
blochsphereplot!(ax, rho)
save("bloch-mixed.png", fig) #hide
nothing #hide
```

![Mixed state inside the Bloch sphere](bloch-mixed.png)

### Dark theme and overlays

Recipe themes use the names `BlochSpherePlot`, `FockDistributionPlot`, `WignerPlot`,
and `WaveFunctionPlot`, following [Makie's theming conventions](https://docs.makie.org/stable/explanations/theming/themes.html).
The sphere attributes can be themed independently of the arrow. This example
uses a coarser mesh and lighter colors. Set `spherevisible=false` when adding an
arrow to an existing sphere.

```@example visualization
fig = with_theme(theme_dark(); BlochSpherePlot=(spherecolor=(:gray65, 0.2),
        wireframecolor=(:gray80, 0.6), wireframewidth=1.5,
        sphereresolution=(16, 8), color=:white)) do
    b = SpinBasis(1//2)
    psi = (spinup(b) + im * spindown(b)) / sqrt(2)
    fig, ax, plot = blochsphereplot(psi;
        axis=(title="Pure spins along y and z", xlabel="⟨σx⟩", ylabel="⟨σy⟩", zlabel="⟨σz⟩"),
        figure=(size=(640, 480),))
    blochsphereplot!(ax, spinup(b); spherevisible=false, color=:orange)
    fig
end
save("bloch-dark.png", fig) #hide
nothing #hide
```

![Two state arrows with a dark theme](bloch-dark.png)

## Fock distributions

The bars show the occupation probabilities ``P(n)=|\langle n|\psi\rangle|^2``
for a ket or ``P(n)=\rho_{nn}`` for a density operator. The horizontal coordinates
are the occupation numbers, including any basis offset.

### Coherent state

```@example visualization
b = FockBasis(20)
fig, ax, plot = fockdistributionplot(coherentstate(b, 2);
    axis=(title="Coherent state, α = 2", xlabel="Occupation n", ylabel="P(n)"),
    figure=(size=(640, 400),))
save("fock-coherent.png", fig) #hide
nothing #hide
```

![Poisson distribution of a coherent state](fock-coherent.png)

### Basis with an offset

```@example visualization
b = FockBasis(12, 4)
fig, ax, plot = fockdistributionplot(fockstate(b, 7);
    axis=(title="Number state in a basis starting at n = 4",
        xlabel="Occupation n", ylabel="P(n)", xticks=4:12),
    figure=(size=(640, 400),))
save("fock-offset.png", fig) #hide
nothing #hide
```

![Number-state distribution with nonzero basis offset](fock-offset.png)

### Comparing pure and mixed states

Makie's bar offsets and color cycling also work with the recipe. The legend uses
the plots' `label` attributes.

```@example visualization
b = FockBasis(30)
psi = coherentstate(b, 2)
rho = 0.5dm(coherentstate(b, 1)) + 0.5dm(coherentstate(b, 3))
fig = Figure(size=(640, 400))
ax = Axis(fig[1, 1]; title="Coherent state and a classical mixture",
    xlabel="Occupation n", ylabel="P(n)", limits=(-0.5, 20.5, nothing, nothing))
fockdistributionplot!(ax, psi; dodge=1, n_dodge=2, label="α = 2")
fockdistributionplot!(ax, rho; dodge=2, n_dodge=2, label="Mixture of α = 1 and α = 3")
axislegend(ax)
save("fock-comparison.png", fig) #hide
nothing #hide
```

![Side-by-side pure and mixed Fock distributions](fock-comparison.png)

## Wigner functions

Pass explicit position and momentum coordinates. The convention is
``\alpha=(x+ip)/\sqrt{2}``; the vacuum has ``W(0,0)=1/\pi``.
The default colormap is `:RdBu`, with red for negative values and blue for positive
values. Automatic limits are `(-m, m)`, where `m` is the largest absolute value on
the plotted grid. Zero therefore stays at the center of the color scale, including
for states with an entirely positive Wigner function. All-zero data use `(-1, 1)`.
Add a `Colorbar` with the returned plot to show that range. The limits update when
the state or grid changes; an explicit `colorrange` takes precedence.

### Coherent state with defaults

```@example visualization
b = FockBasis(30)
x = range(-4, 5; length=151)
p = range(-4, 4; length=151)
fig, ax, plot = wignerplot(coherentstate(b, 1 + 0.5im), x, p;
    axis=(title="Coherent-state Wigner function", xlabel="Position x", ylabel="Momentum p", aspect=DataAspect()),
    figure=(size=(640, 480),))
Colorbar(fig[1, 2], plot; label="W(x, p)")
save("wigner-coherent.png", fig) #hide
nothing #hide
```

![Coherent-state Wigner function with the default colormap](wigner-coherent.png)

### Negative values and a fixed color range

The default diverging colormap and symmetric limits make negative regions easy to
identify. Use a fixed `colorrange` in a theme when comparing different states.

```@example visualization
fig = with_theme(Theme(WignerPlot=(colorrange=(-1/pi, 1/pi),))) do
    b = FockBasis(10)
    x = p = range(-4, 4; length=151)
    fig, ax, plot = wignerplot(fockstate(b, 1), x, p;
        axis=(title="Single-photon Wigner function", xlabel="Position x", ylabel="Momentum p", aspect=DataAspect()),
        figure=(size=(640, 480),))
    Colorbar(fig[1, 2], plot; label="W(x, p)")
    fig
end
save("wigner-negative.png", fig) #hide
nothing #hide
```

![Negative central region of the single-photon Wigner function](wigner-negative.png)

### Cat-state density operator

```@example visualization
b = FockBasis(30)
psi = normalize(coherentstate(b, 2) + coherentstate(b, -2))
x = range(-5, 5; length=201)
p = range(-3, 3; length=151)
fig = Figure(size=(640, 440))
ax = Axis(fig[1, 1]; title="Even cat state", xlabel="Position x", ylabel="Momentum p", aspect=DataAspect())
plot = wignerplot!(ax, dm(psi), x, p; colorrange=(-1/pi, 1/pi))
Colorbar(fig[1, 2], plot; label="W(x, p)")
save("wigner-cat.png", fig) #hide
nothing #hide
```

![Interference fringes of the even cat-state Wigner function](wigner-cat.png)

## Wavefunctions

The recipe converts the discrete coefficients to a sampled wavefunction,
``\psi(q_i)=\langle q_i|\psi\rangle/\sqrt{\Delta q}``, where ``q`` is position or
momentum. It then applies `component` to each sample. The default, `abs2`, shows
probability density, so the area ``\sum_i |\psi(q_i)|^2\Delta q`` is the ket's
squared norm. Use `real`, `imag`, or `abs` to show other components, and label the
vertical axis for the chosen quantity.

### Position probability density

```@example visualization
b = PositionBasis(-6, 6, 256)
psi = gaussianstate(b, 1, 2, 1)
fig, ax, plot = wavefunctionplot(psi;
    axis=(title="Gaussian position density", xlabel="Position x", ylabel="|ψ(x)|²"),
    figure=(size=(640, 400),))
save("wavefunction-position.png", fig) #hide
nothing #hide
```

![Position density of a Gaussian wavepacket](wavefunction-position.png)

### Real and imaginary parts

Both lines use Makie's normal color cycle. Amplitudes and probability densities
have different units; display them on separate axes when comparing them.

```@example visualization
b = PositionBasis(-6, 6, 256)
psi = gaussianstate(b, 0, 3, 1.5)
fig = Figure(size=(640, 400))
ax = Axis(fig[1, 1]; title="Complex Gaussian wavefunction",
    xlabel="Position x", ylabel="Wavefunction amplitude")
wavefunctionplot!(ax, psi; component=real, label="Re ψ(x)")
wavefunctionplot!(ax, psi; component=imag, label="Im ψ(x)")
axislegend(ax)
save("wavefunction-components.png", fig) #hide
nothing #hide
```

![Real and imaginary wavefunction components with default color cycling](wavefunction-components.png)

### Momentum probability density

```@example visualization
bx = PositionBasis(-32, 32, 512)
bp = MomentumBasis(bx)
psi = transform(bp, bx) * gaussianstate(bx, 0, 2, 1)
fig, ax, plot = wavefunctionplot(psi;
    axis=(title="Gaussian momentum density", xlabel="Momentum p", ylabel="|ψ(p)|²", limits=(-3, 7, nothing, nothing)),
    figure=(size=(640, 400),))
save("wavefunction-momentum.png", fig) #hide
nothing #hide
```

![Momentum density obtained by Fourier transformation](wavefunction-momentum.png)
