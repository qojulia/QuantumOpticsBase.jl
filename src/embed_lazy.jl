
# src/embed_lazy.jl  –  Lazy embedding API for QuantumOpticsBase.jl
# Adds a public `embed_lazy` function that embeds local operators into a
# composite Hilbert space WITHOUT materialising the full dense or sparse
# matrix.  Returns LazyTensor / LazySum / TimeDependentSum as appropriate.
# Supported dispatch:
#   embed_lazy(b_l, b_r, index::Integer,            op::AbstractOperator)  → LazyTensor
#   embed_lazy(b_l, b_r, indices::AbstractVector,   ops::AbstractVector)   → LazyTensor
#   embed_lazy(b_l, b_r, indices::AbstractVector,   op::LazyTensor)        → LazyTensor
#   embed_lazy(b_l, b_r, index::Integer,            op::LazyTensor)        → LazyTensor
#   embed_lazy(b_l, b_r, indices,                   op::LazySum)           → LazySum
#   embed_lazy(b_l, b_r, indices,                   op::TimeDependentSum)  → TimeDependentSum
#
#   All six forms also accept a single `basis::CompositeBasis` instead of the
#   (basis_l, basis_r) pair; it is expanded to (basis, basis).

# Validate a single-index single-operator embedding.
function _el_check_single(b_l::CompositeBasis, b_r::CompositeBasis,
                           index::Integer, op::AbstractOperator)
    N = length(b_l.bases)
    (1 <= index <= N) || throw(BoundsError(b_l.bases, index))
    op.basis_l == b_l.bases[index] || throw(IncompatibleBases())
    op.basis_r == b_r.bases[index] || throw(IncompatibleBases())
    nothing
end

# Validate a parallel (indices, ops) pair.
# `ops` may be a Tuple or AbstractVector of AbstractOperator.
function _el_check_multi(b_l::CompositeBasis, b_r::CompositeBasis,
                          indices, ops)
    N  = length(b_l.bases)
    nk = length(indices)
    length(ops) == nk || throw(ArgumentError(
        "embed_lazy: length(indices) = $nk but got $(length(ops)) operators; " *
        "they must be equal (one operator per index)"))
    issorted(indices) || throw(ArgumentError(
        "embed_lazy: `indices` must be sorted in ascending order, got: $indices"))
    for (k, i) in enumerate(indices)
        (1 <= i <= N) || throw(BoundsError(b_l.bases, i))
        ops[k].basis_l == b_l.bases[i] || throw(IncompatibleBases())
        ops[k].basis_r == b_r.bases[i] || throw(IncompatibleBases())
    end
    nothing
end



"""
    embed_lazy(basis_l, basis_r, index::Integer, op::AbstractOperator) -> LazyTensor
    embed_lazy(basis,            index::Integer, op::AbstractOperator) -> LazyTensor

Embed the local operator `op` acting on sub-system `index` into the composite
Hilbert space described by `basis_l`/`basis_r`, returning a [`LazyTensor`](@ref)
**without materialising** the full ``\\prod_i d_i \\times \\prod_i d_i`` matrix.

`op.basis_l` must equal `basis_l.bases[index]` (and similarly for the right
basis), otherwise an `IncompatibleBases` error is thrown.

This is the allocation-free alternative to [`embed`](@ref): `embed` constructs
the full sparse matrix immediately, whereas `embed_lazy` stores only `op` and
its site index, deferring all linear-algebra work until the lazy operator is
applied to a state or explicitly converted via [`dense`](@ref) or
[`sparse`](@ref).

# Examples
```julia
b0 = SpinBasis(1//2)
b  = tensor(b0, b0, b0, b0)     # 4-site chain, dim 16

op_lazy  = embed_lazy(b, 2, sigmax(b0))    # no 16×16 matrix allocated
op_eager = embed(b, 2, sigmax(b0))         # builds 16×16 sparse matrix

dense(op_lazy) ≈ dense(op_eager)           # true

ψ = Ket(b, normalize!(ones(ComplexF64, 16)))
op_lazy * ψ ≈ op_eager * ψ               # true
```

See also [`embed`](@ref), [`LazyTensor`](@ref), [`LazySum`](@ref).
"""
function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
                    index::Integer, op::AbstractOperator)
    _el_check_single(b_l, b_r, index, op)
    LazyTensor(b_l, b_r, [index], (op,))
end

embed_lazy(b::CompositeBasis, index::Integer, op::AbstractOperator) =
    embed_lazy(b, b, index, op)

# AbstractVector of operators: one per site 

"""
    embed_lazy(basis_l, basis_r, indices, ops::AbstractVector{<:AbstractOperator}) -> LazyTensor
    embed_lazy(basis,            indices, ops::AbstractVector{<:AbstractOperator}) -> LazyTensor

Embed a list of single-site operators—one per index—into the composite basis,
returning a single [`LazyTensor`](@ref).  `indices` must be sorted; `ops[k]`
must act on `basis_l.bases[indices[k]]`.

# Example
```julia
b0 = SpinBasis(1//2)
b  = tensor(b0, b0, b0, b0)

lt = embed_lazy(b, [1, 3], [sigmax(b0), sigmaz(b0)])  # sx at site 1, sz at site 3
lt isa LazyTensor   # true
```
"""
function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
                    indices::AbstractVector{Int},
                    ops::AbstractVector{<:AbstractOperator})
    _el_check_multi(b_l, b_r, indices, ops)
    LazyTensor(b_l, b_r, collect(indices), Tuple(ops))
end

embed_lazy(b::CompositeBasis,
           indices::AbstractVector{Int},
           ops::AbstractVector{<:AbstractOperator}) =
    embed_lazy(b, b, indices, ops)

#  LazyTensor re-embedding

"""
    embed_lazy(basis_l, basis_r, indices, op::LazyTensor) -> LazyTensor
    embed_lazy(basis,            indices, op::LazyTensor) -> LazyTensor
    embed_lazy(basis,            index::Integer, op::LazyTensor) -> LazyTensor

Re-embed an existing [`LazyTensor`](@ref) into a larger composite basis.

`indices[k]` specifies the global site (in `basis_l`/`basis_r`) at which
`op.operators[k]`—the k-th non-identity sub-operator stored in `op`—should
be placed.  The scalar `op.factor` is preserved.

`length(indices)` must equal `length(op.operators)`.

# Example
```julia
b0 = SpinBasis(1//2)
b  = tensor(b0, b0, b0, b0)

# LazyTensor on a 2-site sub-basis
lt_sub = LazyTensor(b0 ⊗ b0, [1, 2], (sigmax(b0), sigmaz(b0)))

# Place sx at global site 1, sz at global site 3
lt_big = embed_lazy(b, [1, 3], lt_sub)
lt_big isa LazyTensor   # true
dense(lt_big) ≈ dense(LazyTensor(b, [1, 3], (sigmax(b0), sigmaz(b0))))  # true
```
"""
function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
                    indices::AbstractVector{Int}, op::LazyTensor)
    _el_check_multi(b_l, b_r, indices, op.operators)
    LazyTensor(b_l, b_r, collect(indices), op.operators, op.factor)
end

embed_lazy(b::CompositeBasis, indices::AbstractVector{Int}, op::LazyTensor) =
    embed_lazy(b, b, indices, op)

# Single-index convenience: valid only when op stores exactly one sub-operator.
embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
           index::Integer, op::LazyTensor) =
    embed_lazy(b_l, b_r, [index], op)

embed_lazy(b::CompositeBasis, index::Integer, op::LazyTensor) =
    embed_lazy(b, b, index, op)

#  LazySum 

"""
    embed_lazy(basis_l, basis_r, indices, op::LazySum) -> LazySum
    embed_lazy(basis,            indices, op::LazySum) -> LazySum

Embed a [`LazySum`](@ref) into a larger composite basis, returning a new
`LazySum` whose terms are lazily embedded via `embed_lazy`.  The coefficient
`op.factors` vector is preserved; no sub-operator matrix is materialised.

This is the recommended API for local Hamiltonians.  For instance, a single-site
Hamiltonian ``h = c_1 O_1 + c_2 O_2 + \\ldots`` embeds to a `LazySum` of
`LazyTensor`s:

```julia
b0 = SpinBasis(1//2)
b  = tensor(b0, b0, b0, b0)

local_h = sigmax(b0) + 0.5 * sigmaz(b0)   # LazySum on b0
lazy_h  = embed_lazy(b, 3, local_h)        # LazySum of LazyTensors on b

lazy_h isa LazySum                         # true
dense(lazy_h) ≈ dense(embed(b, 3, local_h))  # true

# Full spin-chain Hamiltonian with zero large-matrix allocations:
H = sum(embed_lazy(b, i, local_h) for i in 1:4)
```
"""
function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis, indices, op::LazySum)
    embedded_ops = map(o -> embed_lazy(b_l, b_r, indices, o), op.operators)
    LazySum(b_l, b_r, op.factors, embedded_ops)
end

embed_lazy(b::CompositeBasis, index::Integer, op::LazySum) =
    embed_lazy(b, b, index, op)

embed_lazy(b::CompositeBasis, indices, op::LazySum) =
    embed_lazy(b, b, indices, op)

# 4-arg Integer+LazySum: disambiguates from (Integer, AbstractOperator)
function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
                    index::Integer, op::LazySum)
    embedded_ops = map(o -> embed_lazy(b_l, b_r, index, o), op.operators)
    LazySum(b_l, b_r, op.factors, embedded_ops)
end

#  TimeDependentSum

"""
    embed_lazy(basis_l, basis_r, indices, op::TimeDependentSum) -> TimeDependentSum
    embed_lazy(basis,            indices, op::TimeDependentSum) -> TimeDependentSum

Embed a [`TimeDependentSum`](@ref) lazily, preserving the time-dependent
coefficient functions and the current clock time.  The underlying `LazySum`
is embedded via `embed_lazy`; coefficient functions are not evaluated.

The returned operator is immediately usable with time-evolution solvers.

```julia
b0 = SpinBasis(1//2)
b  = tensor(b0, b0, b0, b0)

tds = TimeDependentSum([t -> cos(t), t -> sin(t)], (sigmax(b0), sigmaz(b0)))
lazy_tds = embed_lazy(b, 2, tds)
lazy_tds isa TimeDependentSum   # true
```
"""
function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
                    indices, op::TimeDependentSum)
    embedded_lazysum = embed_lazy(b_l, b_r, indices, op.static_op)
    TimeDependentSum(op.coefficients, embedded_lazysum, current_time(op))
end

function embed_lazy(b_l::CompositeBasis, b_r::CompositeBasis,
                    index::Integer, op::TimeDependentSum)
    embedded_lazysum = embed_lazy(b_l, b_r, index, op.static_op)
    TimeDependentSum(op.coefficients, embedded_lazysum, current_time(op))
end

embed_lazy(b::CompositeBasis, index::Integer, op::TimeDependentSum) =
    embed_lazy(b, b, index, op)

embed_lazy(b::CompositeBasis, indices, op::TimeDependentSum) =
    embed_lazy(b, b, indices, op)