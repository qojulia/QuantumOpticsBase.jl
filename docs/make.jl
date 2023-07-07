using Documenter
using QuantumInterface
using QuantumOpticsBase

pages = [
        "index.md",
        "api.md"
    ]

makedocs(
    sitename = "QuantumOpticsBase.jl",
    modules = [QuantumOpticsBase, QuantumInterface],
    pages = pages,
    checkdocs=:exports
    )

deploydocs(
    repo = "github.com/qojulia/QuantumOpticsBase.jl.git",
    )
