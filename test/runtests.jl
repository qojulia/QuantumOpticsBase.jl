names = [
    "test_sortedindices.jl",

    "test_bases.jl",
    "test_states.jl",

    "test_operators.jl",
    "test_operators_dense.jl",
    "test_sparsematrix.jl",
    "test_operators_sparse.jl",
    "test_operators_lazytensor.jl",
    "test_operators_lazysum.jl",
    "test_operators_lazyproduct.jl",
    "test_time_dependent_operators.jl",

    "test_fock.jl",
    "test_spin.jl",
    "test_particle.jl",
    "test_manybody.jl",
    "test_nlevel.jl",
    "test_subspace.jl",
    "test_state_definitions.jl",

    "test_transformations.jl",

    "test_metrics.jl",
    "test_embed.jl",

    "test_superoperators.jl",

    "test_pauli.jl",
    "test_spinors.jl",

    "test_abstractdata.jl",

    "test_printing.jl",

    "test_apply.jl",

    "test_aqua.jl",
    "test_jet.jl"
]

detected_tests = filter(
    name->startswith(name, "test_") && endswith(name, ".jl"),
    readdir("."))

unused_tests = setdiff(detected_tests, names)
if length(unused_tests) != 0
    error("The following tests are not used:\n", join(unused_tests, "\n"))
end

unavailable_tests = setdiff(names, detected_tests)
if length(unavailable_tests) != 0
    error("The following tests could not be found:\n", join(unavailable_tests, "\n"))
end

for name=names
    if startswith(name, "test_") && endswith(name, ".jl")
        include(name)
    end
end
