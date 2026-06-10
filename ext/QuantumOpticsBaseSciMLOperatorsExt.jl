module QuantumOpticsBaseSciMLOperatorsExt

import Base: copy, size, eltype, +, -, *, /
import LinearAlgebra: mul!, adjoint, I

import QuantumOpticsBase
import QuantumOpticsBase: AbstractOperator, Basis, CompositeBasis, DataOperator, DenseOperator,
    SparseOperator, LazySum, LazyProduct, LazyTensor, Ket, Bra, Operator, basis, dense, sparse,
    dagger, identityoperator, suboperator, check_samebases, sciml_lazy_operator

import SciMLOperators
using SciMLOperators: MatrixOperator, TensorProductOperator, cache_operator

struct SciMLLazyOperator{BL,BR,O} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    op::O
end

QuantumOpticsBase.basis(op::SciMLLazyOperator) = op.basis_l

Base.copy(op::SciMLLazyOperator) = SciMLLazyOperator(op.basis_l, op.basis_r, copy(op.op))
Base.eltype(op::SciMLLazyOperator) = eltype(op.op)
Base.size(op::SciMLLazyOperator) = size(op.op)
Base.size(op::SciMLLazyOperator, d::Int) = size(op.op, d)

dense(op::SciMLLazyOperator) = DenseOperator(op.basis_l, op.basis_r, Matrix(op.op))
sparse(op::SciMLLazyOperator) = SparseOperator(op.basis_l, op.basis_r, sparse(op.op))

function _sciml_identity(b1::Basis, b2::Basis, T)
    MatrixOperator(Matrix{T}(I, length(b1), length(b2)))
end

_sciml_raw(op::SciMLLazyOperator) = op.op
_sciml_raw(op::DataOperator) = MatrixOperator(op.data)
_sciml_raw(op::AbstractOperator) = MatrixOperator(dense(op).data)

function _sciml_raw(op::LazySum)
    if isempty(op.operators)
        T = eltype(op.factors)
        return MatrixOperator(zeros(T, length(op.basis_l), length(op.basis_r)))
    end
    terms = map(i -> op.factors[i] * _sciml_raw(op.operators[i]), eachindex(op.operators))
    return reduce(+, terms)
end

function _sciml_raw(op::LazyProduct)
    if isempty(op.operators)
        T = eltype(op)
        return MatrixOperator(Matrix{T}(I, length(op.basis_l), length(op.basis_r)))
    end
    return reduce(*, (_sciml_raw(o) for o in op.operators))
end

function _sciml_raw(op::LazyTensor)
    raw_ops = Any[]
    sizehint!(raw_ops, length(op.basis_l.bases))
    for idx in 1:length(op.basis_l.bases)
        term = findfirst(isequal(idx), op.indices)
        if term === nothing
            push!(raw_ops, _sciml_identity(op.basis_l.bases[idx], op.basis_r.bases[idx], eltype(op)))
        else
            push!(raw_ops, _sciml_raw(suboperator(op, idx)))
        end
    end
    raw = TensorProductOperator(reverse(raw_ops)...)
    return isone(op.factor) ? raw : op.factor * raw
end

sciml_lazy_operator(op::SciMLLazyOperator) = op
sciml_lazy_operator(op::LazySum) = SciMLLazyOperator(op.basis_l, op.basis_r, _sciml_raw(op))
sciml_lazy_operator(op::LazyProduct) = SciMLLazyOperator(op.basis_l, op.basis_r, _sciml_raw(op))
sciml_lazy_operator(op::LazyTensor) = SciMLLazyOperator(op.basis_l, op.basis_r, _sciml_raw(op))
sciml_lazy_operator(op::DataOperator) = SciMLLazyOperator(op.basis_l, op.basis_r, _sciml_raw(op))
sciml_lazy_operator(op::AbstractOperator) = SciMLLazyOperator(op.basis_l, op.basis_r, _sciml_raw(op))

function SciMLOperators.cache_operator(op::SciMLLazyOperator, sample)
    SciMLLazyOperator(op.basis_l, op.basis_r, cache_operator(op.op, sample))
end

function mul!(result::Ket{B1}, op::SciMLLazyOperator{B1,B2}, state::Ket{B2}, alpha, beta) where {B1,B2}
    cached = cache_operator(op.op, state.data)
    mul!(result.data, cached, state.data, alpha, beta)
    return result
end

function mul!(result::Bra{B2}, state::Bra{B1}, op::SciMLLazyOperator{B1,B2}, alpha, beta) where {B1,B2}
    cached = cache_operator(adjoint(op.op), state.data)
    mul!(result.data, cached, state.data, alpha, beta)
    return result
end

function mul!(result::Operator{B1,B3}, op::SciMLLazyOperator{B1,B2}, rhs::Operator{B2,B3}, alpha, beta) where {B1,B2,B3}
    mul!(result.data, dense(op).data, rhs.data, alpha, beta)
    return result
end

function mul!(result::Operator{B1,B3}, lhs::Operator{B1,B2}, op::SciMLLazyOperator{B2,B3}, alpha, beta) where {B1,B2,B3}
    mul!(result.data, lhs.data, dense(op).data, alpha, beta)
    return result
end

function +(a::SciMLLazyOperator{B1,B2}, b::SciMLLazyOperator{B1,B2}) where {B1,B2}
    check_samebases(a, b)
    SciMLLazyOperator(a.basis_l, a.basis_r, a.op + b.op)
end

function +(a::SciMLLazyOperator, b::AbstractOperator)
    check_samebases(a, b)
    SciMLLazyOperator(a.basis_l, a.basis_r, a.op + _sciml_raw(b))
end

function +(a::AbstractOperator, b::SciMLLazyOperator)
    check_samebases(a, b)
    SciMLLazyOperator(a.basis_l, a.basis_r, _sciml_raw(a) + b.op)
end

function -(a::SciMLLazyOperator{B1,B2}, b::SciMLLazyOperator{B1,B2}) where {B1,B2}
    check_samebases(a, b)
    SciMLLazyOperator(a.basis_l, a.basis_r, a.op - b.op)
end

function -(a::SciMLLazyOperator, b::AbstractOperator)
    check_samebases(a, b)
    SciMLLazyOperator(a.basis_l, a.basis_r, a.op - _sciml_raw(b))
end

function -(a::AbstractOperator, b::SciMLLazyOperator)
    check_samebases(a, b)
    SciMLLazyOperator(a.basis_l, a.basis_r, _sciml_raw(a) - b.op)
end

-(a::SciMLLazyOperator) = SciMLLazyOperator(a.basis_l, a.basis_r, -a.op)

function *(a::SciMLLazyOperator{B1,B2}, b::SciMLLazyOperator{B2,B3}) where {B1,B2,B3}
    check_samebases(a.basis_r, b.basis_l)
    SciMLLazyOperator(a.basis_l, b.basis_r, a.op * b.op)
end

function *(a::SciMLLazyOperator, b::AbstractOperator)
    check_samebases(a.basis_r, b.basis_l)
    SciMLLazyOperator(a.basis_l, b.basis_r, a.op * _sciml_raw(b))
end

function *(a::AbstractOperator, b::SciMLLazyOperator)
    check_samebases(a.basis_r, b.basis_l)
    SciMLLazyOperator(a.basis_l, b.basis_r, _sciml_raw(a) * b.op)
end

*(a::SciMLLazyOperator, b::Number) = SciMLLazyOperator(a.basis_l, a.basis_r, a.op * b)
*(a::Number, b::SciMLLazyOperator) = SciMLLazyOperator(b.basis_l, b.basis_r, a * b.op)
/(a::SciMLLazyOperator, b::Number) = SciMLLazyOperator(a.basis_l, a.basis_r, a.op / b)

dagger(op::SciMLLazyOperator) = SciMLLazyOperator(op.basis_r, op.basis_l, adjoint(op.op))

function tensor(a::SciMLLazyOperator, b::SciMLLazyOperator)
    SciMLLazyOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r),
                      TensorProductOperator(b.op, a.op))
end

function tensor(a::SciMLLazyOperator, b::AbstractOperator)
    SciMLLazyOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r),
                      TensorProductOperator(_sciml_raw(b), a.op))
end

function tensor(a::AbstractOperator, b::SciMLLazyOperator)
    SciMLLazyOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r),
                      TensorProductOperator(b.op, _sciml_raw(a)))
end

identityoperator(::Type{<:SciMLLazyOperator}, ::Type{S}, b1::Basis, b2::Basis) where {S<:Number} =
    sciml_lazy_operator(identityoperator(S, b1, b2))

end # module
