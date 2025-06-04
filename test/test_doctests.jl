@testitem "Doctests" tags=[:doctests] begin
    using Documenter
    using QuantumOpticsBase
    using QuantumInterface

    DocMeta.setdocmeta!(QuantumOpticsBase, :DocTestSetup, :(using QuantumOpticsBase); recursive=true)
    modules = [QuantumOpticsBase, QuantumInterface]
    doctestfilters = [r"(QuantumOpticsBase\.|)"]
    doctest(nothing, modules;
            doctestfilters
            #fix=true
           )
    # TODO failures in VSCode related to https://github.com/julia-vscode/TestItemRunner.jl/issues/49
end
