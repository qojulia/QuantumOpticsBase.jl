"""
    embed_lazy([basis_l[, basis_r]], indices, op)

Lazily embed `op` into the composite Hilbert space described by `basis_l`
(and `basis_r`), returning a structure-preserving lazy operator instead of the
eagerly materialized matrix produced by [`embed`](@ref).

In contrast to [`embed`](@ref), which allocates the full embedded matrix (e.g. a
`2^n x 2^n` sparse matrix when embedding a one-site operator in an `n`-site
basis), `embed_lazy` keeps the local tensor-product structure:

- a `DataOperator` (or any other single-subsystem `AbstractOperator`) embedded at
  a single index is returned as a [`LazyTensor`](@ref),
- a [`LazyTensor`](@ref) is re-embedded into the larger basis, preserving its
  suboperators and factor,
- a [`LazySum`](@ref) is preserved as a `LazySum` whose terms are embedded lazily,
- a [`TimeDependentSum`](@ref) is preserved, keeping its coefficients while
  lazily embedding its static operator terms.

The result can be multiplied by kets, used as a term inside a `LazySum`, and
converted to a dense or sparse operator with `dense(...)`/`sparse(...)` only when
explicitly requested.

If both bases are given they must be `CompositeBasis`es; `indices` must be sorted,
mirroring the requirements of [`LazyTensor`](@ref).

A monolithic operator acting jointly on more than one subsystem (for example a
two-qubit gate given as a single dense matrix) has no tensor-product structure to
exploit and therefore cannot be embedded lazily. Such inputs raise an
`ArgumentError`; use [`embed`](@ref) for an eager result, or pass the operator as
a [`LazyTensor`](@ref)/[`LazySum`](@ref) with explicit local structure.

# Examples
```jldoctest
julia> b0 = SpinBasis(1//2); b = b0 ⊗ b0 ⊗ b0 ⊗ b0;

julia> embed_lazy(b, 2, sigmax(b0)) isa LazyTensor
true

julia> embed_lazy(b, 3, LazySum(sigmax(b0), 0.5*sigmaz(b0))) isa LazySum
true
```

See also [`embed`](@ref), [`LazyTensor`](@ref), [`LazySum`](@ref).
"""
function embed_lazy end

embed_lazy(basis::CompositeBasis, indices, op::AbstractOperator) =
    embed_lazy(basis, basis, indices, op)
embed_lazy(basis::CompositeBasis, index::Integer, op::AbstractOperator) =
    embed_lazy(basis, basis, index, op)
embed_lazy(basis::CompositeBasis, indices, op::TimeDependentSum) =
    embed_lazy(basis, basis, indices, op)
embed_lazy(basis::CompositeBasis, index::Integer, op::TimeDependentSum) =
    embed_lazy(basis, basis, index, op)

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, op::AbstractOperator)
    basis_l.bases[index] == op.basis_l || throw(IncompatibleBases())
    basis_r.bases[index] == op.basis_r || throw(IncompatibleBases())
    return LazyTensor(basis_l, basis_r, index, op)
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, op::AbstractOperator)
    if length(indices) == 1
        return embed_lazy(basis_l, basis_r, first(indices), op)
    end
    throw(ArgumentError(
        "embed_lazy cannot lazily embed an operator of type $(typeof(op)) acting " *
        "jointly on the subsystems $(collect(indices)): it has no tensor-product " *
        "structure to preserve. Use `embed` for an eager result, or pass a " *
        "LazyTensor/LazySum with explicit local structure."))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, op::LazyTensor)
    idx = collect(indices)
    issorted(idx) || throw(ArgumentError("embed_lazy requires sorted `indices`."))
    reduce(tensor, basis_l.bases[idx]) == op.basis_l || throw(IncompatibleBases())
    reduce(tensor, basis_r.bases[idx]) == op.basis_r || throw(IncompatibleBases())
    # op.indices address the sub-basis spanned by `indices`; map them to the full basis
    new_indices = idx[op.indices]
    return LazyTensor(basis_l, basis_r, new_indices, op.operators, op.factor)
end
function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, op::LazyTensor)
    return embed_lazy(basis_l, basis_r, [index], op)
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, op::LazySum)
    LazySum(basis_l, basis_r, op.factors, map(o->embed_lazy(basis_l, basis_r, indices, o), op.operators))
end
function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, op::LazySum)
    LazySum(basis_l, basis_r, op.factors, map(o->embed_lazy(basis_l, basis_r, index, o), op.operators))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, op::TimeDependentSum)
    TimeDependentSum(coefficients(op), embed_lazy(basis_l, basis_r, index, static_operator(op)), op.current_time)
end
function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, op::TimeDependentSum)
    TimeDependentSum(coefficients(op), embed_lazy(basis_l, basis_r, indices, static_operator(op)), op.current_time)
end
