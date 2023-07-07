using Documenter
using QuantumOpticsBase
using QuantumInteface

pages = [
        "index.md",
        "api.md"
    ]

makedocs(
    sitename = "QuantumOpticsBase.jl",
    modules = [QuantumOpticsBase, QuantumInteface],
    pages = pages,
    checkdocs=:exports
    )

deploydocs(
    repo = "github.com/qojulia/QuantumOpticsBase.jl.git",
    )
