using BenchmarkTools
using LinearAlgebra
using QuantumOpticsBase
using Random
using SciMLOperators
using SparseArrays

const SUITE = BenchmarkGroup()

for family in ("lazy_sum", "lazy_product", "lazy_tensor", "mixed_hamiltonian")
    SUITE[family] = BenchmarkGroup([family])
end

function chain_basis(n)
    site = SpinBasis(1//2)
    return tensor(ntuple(_ -> site, n)...)
end

function random_ket(basis, seed)
    rng = MersenneTwister(seed)
    return Ket(basis, randn(rng, ComplexF64, length(basis)))
end

function chain_sum_case(n)
    site = SpinBasis(1//2)
    basis = tensor(ntuple(_ -> site, n)...)
    sx = sigmax(site)
    sz = sigmaz(site)
    terms = AbstractOperator[LazyTensor(basis, i, isodd(i) ? sx : sz) for i in 1:n]
    return LazySum(terms...)
end

function product_case(n)
    site = SpinBasis(1//2)
    basis = tensor(ntuple(_ -> site, n)...)
    sx = sigmax(site)
    sz = sigmaz(site)
    terms = AbstractOperator[LazyTensor(basis, i, isodd(i) ? sx : sz) for i in 1:n]
    return LazyProduct(terms...)
end

function two_site_product_case(n)
    site = SpinBasis(1//2)
    basis = tensor(ntuple(_ -> site, n)...)
    mid = max(1, div(n, 2))
    return LazyProduct(LazyTensor(basis, mid, sigmax(site)), LazyTensor(basis, mid + 1, sigmaz(site)))
end

function tensor_factors(site, storage)
    factors = (sigmax(site), sigmaz(site))
    return storage == :dense ? dense.(factors) : sparse.(factors)
end

function middle_sites(n)
    left = max(2, div(n, 2))
    return (left, left + 1)
end

function tensor_case(n; spin=1//2, storage=:sparse, sites=(1, n))
    site = SpinBasis(spin)
    basis = tensor(ntuple(_ -> site, n)...)
    return LazyTensor(basis, collect(sites), tensor_factors(site, storage))
end

function mixed_hamiltonian_case(n)
    site = SpinBasis(1//2)
    basis = tensor(ntuple(_ -> site, n)...)
    sx = sigmax(site)
    sz = sigmaz(site)
    terms = AbstractOperator[]

    for i in 1:(n - 1)
        push!(terms, LazyProduct(LazyTensor(basis, i, sx), LazyTensor(basis, i + 1, sx)))
        push!(terms, LazyProduct(LazyTensor(basis, i, sz), LazyTensor(basis, i + 1, sz)))
    end

    for i in 1:n
        push!(terms, 0.3 * LazyTensor(basis, i, sz))
    end

    return LazySum(terms...)
end

function add_apply_benchmarks!(suite, name, current, psi)
    group = BenchmarkGroup([name])
    sciml_uncached = sciml_lazy_operator(current)
    sciml_cached = cache_sciml_lazy_operator(sciml_uncached, psi)

    out_current = Ket(current.basis_l)
    out_uncached = Ket(current.basis_l)
    out_cached = Ket(current.basis_l)

    group["current"] = @benchmarkable mul!($out_current, $current, $psi, 1, 0)
    group["sciml_uncached"] = @benchmarkable mul!($out_uncached, $sciml_uncached, $psi, 1, 0)
    group["sciml_cached"] = @benchmarkable mul!($out_cached, $sciml_cached, $psi, 1, 0)

    suite[name] = group
    return group
end

function add_tensor_benchmarks!(n; spin=1//2, storage=:sparse, sites=(1, n), seed=40 + n)
    op = tensor_case(n; spin=spin, storage=storage, sites=sites)
    site_label = sites == (1, n) ? "edge" : "middle"
    spin_label = spin == 1//2 ? "spin=1/2" : "spin=$spin"
    name = "$(String(storage)) $site_label $spin_label n=$n"
    return add_apply_benchmarks!(SUITE["lazy_tensor"], name, op, random_ket(op.basis_r, seed))
end

for n in (4, 6, 8)
    op = chain_sum_case(n)
    add_apply_benchmarks!(SUITE["lazy_sum"], "n=$n", op, random_ket(op.basis_r, n))
end

for n in (4, 6)
    op = product_case(n)
    add_apply_benchmarks!(SUITE["lazy_product"], "n=$n", op, random_ket(op.basis_r, 20 + n))
end

for n in (4, 6, 8)
    op = two_site_product_case(n)
    add_apply_benchmarks!(SUITE["lazy_product"], "two_site n=$n", op, random_ket(op.basis_r, 30 + n))
end

for n in (4, 6, 8)
    for storage in (:sparse, :dense)
        add_tensor_benchmarks!(n; storage=storage, sites=(1, n), seed=40 + n + (storage == :dense ? 100 : 0))
        add_tensor_benchmarks!(n; storage=storage, sites=middle_sites(n), seed=80 + n + (storage == :dense ? 100 : 0))
    end
end

for n in (3, 4)
    for storage in (:sparse, :dense)
        add_tensor_benchmarks!(n; spin=1, storage=storage, sites=(1, n), seed=120 + n + (storage == :dense ? 100 : 0))
    end
end

for storage in (:sparse, :dense)
    add_tensor_benchmarks!(4; spin=1, storage=storage, sites=middle_sites(4), seed=160 + (storage == :dense ? 100 : 0))
end

for n in (4, 6)
    op = mixed_hamiltonian_case(n)
    add_apply_benchmarks!(SUITE["mixed_hamiltonian"], "n=$n", op, random_ket(op.basis_r, 60 + n))
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run(SUITE; verbose=true)
    show(stdout, MIME("text/plain"), results)
    println()
end
