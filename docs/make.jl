using Documenter
using AnythingLLMDocs
using QuantumInterface
using QuantumOpticsBase

pages = [
        "index.md",
        "api.md"
    ]

doc_modules = [QuantumOpticsBase, QuantumInterface]

api_base="https://anythingllm.krastanov.org/api/v1"
anythingllm_assets = integrate_anythingllm(
    "QuantumOpticsBase",
    doc_modules,
    @__DIR__,
    api_base;
    repo = "github.com/qojulia/QuantumOpticsBase.jl.git",
    options = EmbedOptions(),
)

makedocs(
    sitename = "QuantumOpticsBase.jl",
    modules = doc_modules,
    pages = pages,
    format = Documenter.HTML(assets = anythingllm_assets),
    checkdocs=:exports
    )

deploydocs(
    repo = "github.com/qojulia/QuantumOpticsBase.jl.git",
    )
