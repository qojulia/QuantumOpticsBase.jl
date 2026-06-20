# benchmark/sciml_lazy_benchmark.jl
#
# Comparative benchmark: QuantumOpticsBase native lazy operators
# vs SciMLOperators-backed prototype.
#
# Usage:
#   cd QuantumOpticsBase.jl
#   julia --project=benchmark benchmark/sciml_lazy_benchmark.jl
#
# Output: a Markdown table of minimum times (BenchmarkTools, 200-sample budget).

using BenchmarkTools
using QuantumOpticsBase
using SciMLOperators
using LinearAlgebra
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.samples  = 200
BenchmarkTools.DEFAULT_PARAMETERS.seconds  = 2.0
BenchmarkTools.DEFAULT_PARAMETERS.evals    = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fmt(t) = @sprintf("%.2f μs", t * 1e6)

function run_triple(label, lazy_op, psi)
    w       = sciml_lazy_operator(lazy_op)
    w_cache = cache_sciml_lazy_operator(w, psi.data)

    t_lazy   = @belapsed $lazy_op * $psi
    t_sciml  = @belapsed $w      * $psi
    t_cached = @belapsed $w_cache * $psi

    @printf("| %-45s | %10s | %14s | %13s |\n",
        label, fmt(t_lazy), fmt(t_sciml), fmt(t_cached))
end

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------
println()
println("## SciMLOperators prototype benchmark")
println()
println("Julia $(VERSION), SciMLOperators $(pkgversion(SciMLOperators))")
println()
@printf("| %-45s | %10s | %14s | %13s |\n",
    "case", "native lazy", "SciML uncached", "SciML cached")
println("|" * "-"^47 * "|" * "-"^12 * "|" * "-"^16 * "|" * "-"^15 * "|")

b0 = SpinBasis(1//2)
b1 = SpinBasis(1)
sx, sy, sz = sigmax(b0), sigmay(b0), sigmaz(b0)
sp, sm     = sigmap(b0), sigmam(b0)

# ---------------------------------------------------------------------------
# 1. LazySum: local transverse-field terms on spin chains (n = 4, 6, 8)
# ---------------------------------------------------------------------------
for n in (4, 6, 8)
    b  = tensor(fill(b0, n)...)
    H  = LazySum([LazyTensor(b, k, sx) for k in 1:n]...)
    psi = Ket(b, normalize!(randn(ComplexF64, length(b))))
    run_triple("LazySum  n=$n (transverse field)", H, psi)
end

# ---------------------------------------------------------------------------
# 2. LazyProduct: depth-2, -4, -6 chains
# ---------------------------------------------------------------------------
for depth in (2, 4, 6)
    n  = max(depth, 4)
    b  = tensor(fill(b0, n)...)
    psi = Ket(b, normalize!(randn(ComplexF64, length(b))))
    ops = [LazyTensor(b, mod1(k, n), sx) for k in 1:depth]
    P  = LazyProduct(ops...)
    run_triple("LazyProduct  depth=$depth n=$n", P, psi)
end

# ---------------------------------------------------------------------------
# 3. LazyTensor: edge vs mid, dense vs sparse, spin-½ n=6
# ---------------------------------------------------------------------------
let n = 6, b = tensor(fill(b0, n)...)
    psi = Ket(b, normalize!(randn(ComplexF64, length(b))))

    run_triple("LazyTensor  edge-1  dense  n=$n", LazyTensor(b, 1, sx), psi)
    run_triple("LazyTensor  edge-$n dense  n=$n", LazyTensor(b, n, sz), psi)
    run_triple("LazyTensor  mid-3   dense  n=$n", LazyTensor(b, 3, sy), psi)
    run_triple("LazyTensor  mid-3   sparse n=$n", LazyTensor(b, 3, SparseOperator(sy)), psi)
    run_triple("LazyTensor  edge-1  sparse n=$n", LazyTensor(b, 1, SparseOperator(sx)), psi)
end

# ---------------------------------------------------------------------------
# 4. LazyTensor: spin-1 local dimension (d=3)
# ---------------------------------------------------------------------------
let n = 4, b = tensor(fill(b1, n)...),
    sx1 = sigmax(b1), sz1 = sigmaz(b1)
    psi = Ket(b, normalize!(randn(ComplexF64, length(b))))

    run_triple("LazyTensor  mid-2   spin-1 n=$n", LazyTensor(b, 2, sx1), psi)
    run_triple("LazyTensor  mid-3   spin-1 n=$n", LazyTensor(b, 3, sz1), psi)
end

# ---------------------------------------------------------------------------
# 5. Mixed Heisenberg-like Hamiltonian (nearest-neighbour XX+YY+ZZ)
# ---------------------------------------------------------------------------
for n in (4, 6)
    b  = tensor(fill(b0, n)...)
    psi = Ket(b, normalize!(randn(ComplexF64, length(b))))
    terms = AbstractOperator[]
    for k in 1:(n-1)
        push!(terms, LazyProduct(LazyTensor(b, k, sx), LazyTensor(b, k+1, sx)))
        push!(terms, LazyProduct(LazyTensor(b, k, sy), LazyTensor(b, k+1, sy)))
        push!(terms, LazyProduct(LazyTensor(b, k, sz), LazyTensor(b, k+1, sz)))
    end
    H = LazySum(terms...)
    run_triple("Mixed Heisenberg  n=$n ($(length(terms)) terms)", H, psi)
end

println()
println("> Timings are minimum over $(BenchmarkTools.DEFAULT_PARAMETERS.samples) samples.")
println("> 'SciML uncached' allocates intermediate buffers on every call.")
println("> 'SciML cached'  pre-allocates via `cache_sciml_lazy_operator`.")