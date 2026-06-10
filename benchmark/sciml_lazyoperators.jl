import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using BenchmarkTools
using SciMLOperators
using QuantumOpticsBase
using Random

const RNG = MersenneTwister(20260609)

function spin_chain(n)
    b = SpinBasis(1//2)
    foldl(⊗, fill(b, n))
end

function local_spin_term(chain, idx, op)
    LazyTensor(chain, chain, [idx], (op,))
end

function structured_hamiltonian(n)
    chain = spin_chain(n)
    sx = sigmax(SpinBasis(1//2))
    sz = sigmaz(SpinBasis(1//2))
    terms = Any[]
    for i in 1:n
        push!(terms, local_spin_term(chain, i, isodd(i) ? sx : sz))
    end
    if n >= 2
        push!(terms, LazyProduct(local_spin_term(chain, 1, sx), local_spin_term(chain, 2, sz)))
    end
    LazySum(terms...)
end

function benchmark_group()
    group = BenchmarkGroup()
    group["lazy_sum"] = BenchmarkGroup()
    group["lazy_product"] = BenchmarkGroup()
    group["lazy_tensor"] = BenchmarkGroup()
    group["mixed_hamiltonian"] = BenchmarkGroup()

    for n in (4, 6, 8)
        chain = spin_chain(n)
        ψ = Ket(chain, randn(RNG, ComplexF64, length(chain)))
        H = structured_hamiltonian(n)
        Hs = QuantumOpticsBase.sciml_lazy_operator(H)
        Hsc = cache_operator(Hs, ψ.data)

        group["lazy_sum"][n] = BenchmarkGroup()
        group["lazy_sum"][n]["current"] = @benchmarkable $H * $ψ
        group["lazy_sum"][n]["sciml"] = @benchmarkable $Hs * $ψ
        group["lazy_sum"][n]["sciml_cached"] = @benchmarkable $Hsc * $ψ

        sx = sigmax(SpinBasis(1//2))
        sz = sigmaz(SpinBasis(1//2))

        P = LazyProduct(local_spin_term(chain, 1, sx),
                        local_spin_term(chain, 2, sz))
        Ps = QuantumOpticsBase.sciml_lazy_operator(P)
        Psc = cache_operator(Ps, ψ.data)

        group["lazy_product"][n] = BenchmarkGroup()
        group["lazy_product"][n]["current"] = @benchmarkable $P * $ψ
        group["lazy_product"][n]["sciml"] = @benchmarkable $Ps * $ψ
        group["lazy_product"][n]["sciml_cached"] = @benchmarkable $Psc * $ψ

        T = LazyTensor(chain, chain, [1, n], (sx, sz))
        Ts = QuantumOpticsBase.sciml_lazy_operator(T)
        Tsc = cache_operator(Ts, ψ.data)

        group["lazy_tensor"][n] = BenchmarkGroup()
        group["lazy_tensor"][n]["current"] = @benchmarkable $T * $ψ
        group["lazy_tensor"][n]["sciml"] = @benchmarkable $Ts * $ψ
        group["lazy_tensor"][n]["sciml_cached"] = @benchmarkable $Tsc * $ψ

        group["mixed_hamiltonian"][n] = BenchmarkGroup()
        group["mixed_hamiltonian"][n]["current"] = @benchmarkable $H * $ψ
        group["mixed_hamiltonian"][n]["sciml"] = @benchmarkable $Hs * $ψ
        group["mixed_hamiltonian"][n]["sciml_cached"] = @benchmarkable $Hsc * $ψ
    end

    return group
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run(benchmark_group(); verbose = true)
    display(results)
end
