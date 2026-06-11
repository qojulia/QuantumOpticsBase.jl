module QuantumOpticsBaseSciMLOperatorsExt

using QuantumOpticsBase
using SciMLOperators
import LinearAlgebra
import LinearAlgebra: mul!, I
import QuantumOpticsBase: AbstractOperator, Operator, Ket, Bra,
    LazySum, LazyProduct, LazyTensor, CompositeBasis,
    dense, dagger, check_samebases

export SciMLOperatorWrapper

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

"""
    SciMLOperatorWrapper{BL,BR,T} <: AbstractOperator{BL,BR}

Wraps an `AbstractSciMLOperator` while preserving QuantumOpticsBase basis
information. Construct via [`sciml_lazy_operator`](@ref).
"""
struct SciMLOperatorWrapper{BL,BR,T} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    sciml_op::T
end

Base.eltype(::SciMLOperatorWrapper{BL,BR,T}) where {BL,BR,T} = eltype(T)
Base.size(w::SciMLOperatorWrapper) = (length(w.basis_l), length(w.basis_r))

# ---------------------------------------------------------------------------
# Conversion helpers: QO types → SciML operator trees
# ---------------------------------------------------------------------------

function _qo_to_sciml(op::Operator)
    MatrixOperator(op.data)
end

function _qo_to_sciml(op::AbstractOperator)
    MatrixOperator(dense(op).data)
end

function _qo_to_sciml(op::LazySum)
    if isempty(op.operators)
        T = eltype(op.factors)
        return MatrixOperator(zeros(T, length(op.basis_l), length(op.basis_r)))
    end
    result = op.factors[1] * _qo_to_sciml(op.operators[1])
    for i in 2:length(op.operators)
        result = result + op.factors[i] * _qo_to_sciml(op.operators[i])
    end
    result
end

function _qo_to_sciml(op::LazyProduct)
    isempty(op.operators) &&
        error("Cannot convert an empty LazyProduct to SciMLOperators.")
    result = _qo_to_sciml(op.operators[1])
    for i in 2:length(op.operators)
        result = result * _qo_to_sciml(op.operators[i])
    end
    isone(op.factor) ? result : op.factor * result
end

function _qo_to_sciml(op::LazyTensor)
    cb     = op.basis_l
    nsites = length(cb.bases)
    T      = eltype(op)
    site_ops = map(1:nsites) do k
        idx = findfirst(==(k), op.indices)
        if idx !== nothing
            _qo_to_sciml(op.operators[idx])
        else
            d = length(cb.bases[k])
            MatrixOperator(Matrix{T}(I, d, d))
        end
    end
    # QuantumOpticsBase uses Fortran/column-major ordering: site 1 is the
    # fastest-varying index (stride 1). SciMLOperators' TensorProductOperator
    # uses C/row-major ordering: the first argument is slowest-varying.
    # Reversing aligns the two conventions.
    tp = TensorProductOperator(reverse(site_ops)...)
    isone(op.factor) ? tp : op.factor * tp
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    sciml_lazy_operator(op::AbstractOperator) -> SciMLOperatorWrapper

Convert a QuantumOpticsBase lazy operator to a SciMLOperators-backed wrapper,
preserving `basis_l` and `basis_r`. Requires `SciMLOperators` to be loaded.

```julia
using QuantumOpticsBase, SciMLOperators
b0 = SpinBasis(1//2)
b  = tensor(b0, b0, b0, b0)
H  = LazySum(LazyTensor(b,1,sigmax(b0)), LazyTensor(b,2,sigmaz(b0)))
H_sciml = sciml_lazy_operator(H)
psi = Ket(b, randn(ComplexF64, length(b)))
@assert dense(H_sciml) ≈ dense(H)
@assert H_sciml * psi  ≈ H * psi
```
"""
function QuantumOpticsBase.sciml_lazy_operator(op::AbstractOperator)
    SciMLOperatorWrapper(op.basis_l, op.basis_r, _qo_to_sciml(op))
end

"""
    cache_sciml_lazy_operator(w::SciMLOperatorWrapper, u::AbstractVector)

Pre-allocate intermediate buffers via `SciMLOperators.cache_operator`.
Repeated `mul!` calls on the returned wrapper avoid per-call heap allocation.
"""
function QuantumOpticsBase.cache_sciml_lazy_operator(w::SciMLOperatorWrapper, u::AbstractVector)
    SciMLOperatorWrapper(w.basis_l, w.basis_r, cache_operator(w.sciml_op, u))
end

# ---------------------------------------------------------------------------
# QuantumOpticsBase interface
# ---------------------------------------------------------------------------

function QuantumOpticsBase.dense(w::SciMLOperatorWrapper)
    n   = length(w.basis_l)
    m   = length(w.basis_r)
    dat = zeros(ComplexF64, n, m)
    e_j = zeros(ComplexF64, m)
    col = zeros(ComplexF64, n)
    # TensorProductOperator (and any tree containing it) requires cache before mul!.
    # dense() is O(n^2) allocation anyway, so caching here is fine.
    cached_op = cache_operator(w.sciml_op, e_j)
    for j in 1:m
        e_j[j] = one(ComplexF64)
        mul!(col, cached_op, e_j)
        dat[:, j] .= col
        e_j[j] = zero(ComplexF64)
    end
    Operator(w.basis_l, w.basis_r, dat)
end

function mul!(result::Ket{B1}, op::SciMLOperatorWrapper{B1,B2},
              psi::Ket{B2}, alpha::Number, beta::Number) where {B1,B2}
    cached_op = cache_operator(op.sciml_op, psi.data)
    mul!(result.data, cached_op, psi.data, alpha, beta)
    return result
end

function Base.:*(op::SciMLOperatorWrapper{B1,B2}, psi::Ket{B2}) where {B1,B2}
    out = zeros(eltype(psi), length(op.basis_l))
    cached_op = cache_operator(op.sciml_op, psi.data)
    mul!(out, cached_op, psi.data)
    return Ket(op.basis_l, out)
end

Base.:*(α::Number, op::SciMLOperatorWrapper) =
    SciMLOperatorWrapper(op.basis_l, op.basis_r, α * op.sciml_op)
Base.:*(op::SciMLOperatorWrapper, α::Number) =
    SciMLOperatorWrapper(op.basis_l, op.basis_r, op.sciml_op * α)
Base.:/(op::SciMLOperatorWrapper, α::Number) =
    SciMLOperatorWrapper(op.basis_l, op.basis_r, op.sciml_op / α)

function Base.:+(a::SciMLOperatorWrapper{B1,B2},
                  b::SciMLOperatorWrapper{B1,B2}) where {B1,B2}
    check_samebases(a, b)
    SciMLOperatorWrapper(a.basis_l, a.basis_r, a.sciml_op + b.sciml_op)
end

function Base.:*(a::SciMLOperatorWrapper{B1,B2},
                  b::SciMLOperatorWrapper{B2,B3}) where {B1,B2,B3}
    SciMLOperatorWrapper(a.basis_l, b.basis_r, a.sciml_op * b.sciml_op)
end

function QuantumOpticsBase.dagger(w::SciMLOperatorWrapper)
    d = dense(w)
    SciMLOperatorWrapper(w.basis_r, w.basis_l, MatrixOperator(adjoint(d.data)))
end

end  # module QuantumOpticsBaseSciMLOperatorsExt
