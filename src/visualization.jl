"""
    blochsphere(state::Ket; kwargs...) -> (Figure, Axis3, Plot)

Visualize a pure qubit state on a Bloch sphere using Makie.

The Bloch vector components are the expectation values of the Pauli operators:
x = 2Re(ᾱβ), y = 2Im(ᾱβ), z = |α|² - |β|²

# Keyword Arguments
- `arrowcolor`: color of the state vector arrow (default `:red`)
- `spherecolor`: color of the sphere surface (default `:lightblue`)
- `spherealpha`: transparency of the sphere (default `0.15`)
- `showwireframe`: show equator and meridian circles (default `true`)
- `showaxes`: show dashed x/y/z axis lines (default `true`)
- `showlabels`: show pole labels |0⟩ |1⟩ |+⟩ |-⟩ |+i⟩ |-i⟩ (default `true`)
- `labelsize`: font size for pole labels (default `18`)
- `limits`: axis range, sphere has radius 1 (default `1.6`)

Requires a Makie backend (e.g. `using CairoMakie` or `using GLMakie`).
"""
function blochsphere end