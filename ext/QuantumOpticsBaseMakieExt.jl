module QuantumOpticsBaseMakieExt

import QuantumOpticsBase
import QuantumOpticsBase: Ket, wigner, wignerplot, wignerplot!, wignerplot_axis
using Makie

@recipe(WignerPlot, state) do scene
    Attributes(
        xrange   = (-5.0, 5.0),
        prange   = (-5.0, 5.0),
        npoints  = 100,
        colormap = :RdBu,
    )
end

function Makie.plot!(p::WignerPlot)
    state_obs = p[1]

    grid = @lift begin
        s            = $state_obs
        xmin, xmax   = p[:xrange][]
        pmin, pmax   = p[:prange][]
        n            = p[:npoints][]
        xvec         = collect(LinRange(Float64(xmin), Float64(xmax), n))
        pvec         = collect(LinRange(Float64(pmin), Float64(pmax), n))
        W            = wigner(s, xvec, pvec)
        mx           = max(abs(minimum(W)), abs(maximum(W)))
        (xvec, pvec, W, mx)
    end

    xs   = @lift $grid[1]
    pvs  = @lift $grid[2]
    Ws   = @lift $grid[3]
    clim = @lift (-$grid[4], $grid[4])

    heatmap!(p, xs, pvs, Ws;
        colormap   = p[:colormap],
        colorrange = clim,
    )

    return p
end

function QuantumOpticsBase.wignerplot_axis(ax, state; kwargs...)
    wignerplot!(ax, state; kwargs...)
end

function QuantumOpticsBase.wignerplot_axis(state; kwargs...)
    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1];
    xlabel          = "x",
    ylabel          = "p",
    aspect          = DataAspect(),
    xgridvisible    = false,
    ygridvisible    = false,
    backgroundcolor = :white,
    )
    plt = wignerplot_axis(ax, state; kwargs...)

    Colorbar(fig[1, 2], plt;
        label = "W(x,p)",
        width = 15,
    )

    return fig, ax, plt
end

end # module