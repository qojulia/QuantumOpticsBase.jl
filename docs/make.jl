using Documenter
using QuantumOpticsBase

pages = [
        "index.md",
        "api.md"
    ]

makedocs(
    sitename = "QuantumOpticsBase.jl",
    modules = [QuantumOpticsBase],
    pages = pages,
    checkdocs=:exports
    )
