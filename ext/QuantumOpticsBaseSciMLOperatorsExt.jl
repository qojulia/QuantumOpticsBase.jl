module QuantumOpticsBaseSciMLOperatorsExt

import Base: +, -, *, /, copy, eltype, size, transpose
import LinearAlgebra: adjoint, mul!
import SparseArrays
import QuantumOpticsBase
import QuantumOpticsBase: Bra, Ket
import SciMLOperators

const QOB = QuantumOpticsBase

struct SciMLLazyOperator{BL,BR,O} <: QOB.LazyOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    operator::O

    function SciMLLazyOperator(basis_l::BL, basis_r::BR, operator::O) where {BL,BR,O}
        expected = (length(basis_l), length(basis_r))
        size(operator) == expected || throw(DimensionMismatch("SciML operator has size $(size(operator)), expected $expected from bases."))
        return new{BL,BR,O}(basis_l, basis_r, operator)
    end
end

copy(op::SciMLLazyOperator) = SciMLLazyOperator(op.basis_l, op.basis_r, copy(op.operator))
eltype(op::SciMLLazyOperator) = eltype(op.operator)
size(op::SciMLLazyOperator) = (length(op.basis_l), length(op.basis_r))
size(op::SciMLLazyOperator, dim::Int) = size(op)[dim]
size(op::SciMLLazyOperator, dim::Integer) = size(op)[dim]

function _check_same_bases(a, b)
    a.basis_l == b.basis_l && a.basis_r == b.basis_r || throw(QOB.IncompatibleBases())
    return nothing
end

function _check_multiplicable(a, b)
    a.basis_r == b.basis_l || throw(QOB.IncompatibleBases())
    return nothing
end

_wrap(basis_l, basis_r, operator) = SciMLLazyOperator(basis_l, basis_r, operator)

function _operator_number_type(op)
    T = eltype(op)
    return T <: Number ? T : ComplexF64
end

function _matrix_identity(::Type{T}, basis_l, basis_r) where {T}
    if length(basis_l) == length(basis_r)
        return SciMLOperators.IdentityOperator(length(basis_l))
    end
    return SciMLOperators.MatrixOperator(QOB.identityoperator(T, basis_l, basis_r).data)
end

_sciml_operator(op::SciMLLazyOperator) = op.operator
_sciml_operator(op::QOB.DataOperator) = SciMLOperators.MatrixOperator(op.data)

function _sciml_operator(op::QOB.LazySum)
    if isempty(op.operators)
        return SciMLOperators.NullOperator(length(op.basis_l), length(op.basis_r))
    end

    result = op.factors[1] * _sciml_operator(op.operators[1])
    for i in 2:length(op.operators)
        result = result + op.factors[i] * _sciml_operator(op.operators[i])
    end
    return result
end

function _combined_lazytensor_product(op::QOB.LazyProduct)
    length(op.operators) == 2 || return nothing
    all(operator -> operator isa QOB.LazyTensor, op.operators) || return nothing

    result = op.operators[1]
    for i in 2:length(op.operators)
        result = result * op.operators[i]
        result isa QOB.LazyTensor || return nothing
    end
    return result
end

function _sciml_operator(op::QOB.LazyProduct)
    combined = _combined_lazytensor_product(op)
    combined === nothing || return op.factor * _sciml_operator(combined)

    result = _sciml_operator(op.operators[1])
    for i in 2:length(op.operators)
        result = result * _sciml_operator(op.operators[i])
    end
    return op.factor * result
end

function _sciml_operator(op::QOB.LazyTensor)
    T = _operator_number_type(op)
    factors = map(1:length(op.basis_l.bases)) do index
        op_index = findfirst(isequal(index), op.indices)
        if op_index === nothing
            return _matrix_identity(T, op.basis_l.bases[index], op.basis_r.bases[index])
        end
        return _sciml_operator(op.operators[op_index])
    end

    # QuantumOpticsBase stores tensor data as kron(right, left), which matches
    # SciMLOperators when the factor order is reversed.
    tensor_operator = SciMLOperators.TensorProductOperator(reverse(factors)...)
    return op.factor * tensor_operator
end

function _sciml_operator(op::QOB.AbstractOperator)
    return SciMLOperators.MatrixOperator(QOB.dense(op).data)
end

function QOB.sciml_lazy_operator(op::QOB.AbstractOperator; cache=nothing)
    wrapped = _wrap(op.basis_l, op.basis_r, _sciml_operator(op))
    cache === nothing && return wrapped
    return QOB.cache_sciml_lazy_operator(wrapped, cache)
end

function QOB.sciml_lazy_operator(op::SciMLLazyOperator; cache=nothing)
    cache === nothing && return op
    return QOB.cache_sciml_lazy_operator(op, cache)
end

function QOB.cache_sciml_lazy_operator(op::SciMLLazyOperator, state::Ket)
    return QOB.cache_sciml_lazy_operator(op, state.data)
end

function QOB.cache_sciml_lazy_operator(op::SciMLLazyOperator, data::AbstractVecOrMat)
    return _wrap(op.basis_l, op.basis_r, SciMLOperators.cache_operator(op.operator, data))
end

function QOB.cache_sciml_lazy_operator(op::QOB.AbstractOperator, state)
    return QOB.cache_sciml_lazy_operator(QOB.sciml_lazy_operator(op), state)
end

function QOB.dense(op::SciMLLazyOperator)
    return QOB.DenseOperator(op.basis_l, op.basis_r, Matrix(op.operator))
end

function SparseArrays.sparse(op::SciMLLazyOperator)
    return QOB.SparseOperator(op.basis_l, op.basis_r, SparseArrays.sparse(Matrix(op.operator)))
end

function -(op::SciMLLazyOperator)
    return _wrap(op.basis_l, op.basis_r, -op.operator)
end

function +(a::SciMLLazyOperator, b::SciMLLazyOperator)
    _check_same_bases(a, b)
    return _wrap(a.basis_l, a.basis_r, a.operator + b.operator)
end

function -(a::SciMLLazyOperator, b::SciMLLazyOperator)
    _check_same_bases(a, b)
    return _wrap(a.basis_l, a.basis_r, a.operator - b.operator)
end

function +(a::SciMLLazyOperator, b::QOB.LazyOperator)
    return a + QOB.sciml_lazy_operator(b)
end

function +(a::QOB.LazyOperator, b::SciMLLazyOperator)
    return QOB.sciml_lazy_operator(a) + b
end

function +(a::SciMLLazyOperator, b::QOB.AbstractOperator)
    return a + QOB.sciml_lazy_operator(b)
end

function +(a::QOB.AbstractOperator, b::SciMLLazyOperator)
    return QOB.sciml_lazy_operator(a) + b
end

function -(a::SciMLLazyOperator, b::QOB.AbstractOperator)
    return a - QOB.sciml_lazy_operator(b)
end

function -(a::QOB.AbstractOperator, b::SciMLLazyOperator)
    return QOB.sciml_lazy_operator(a) - b
end

function -(a::SciMLLazyOperator, b::QOB.LazyOperator)
    return a - QOB.sciml_lazy_operator(b)
end

function -(a::QOB.LazyOperator, b::SciMLLazyOperator)
    return QOB.sciml_lazy_operator(a) - b
end

function *(a::SciMLLazyOperator, b::SciMLLazyOperator)
    _check_multiplicable(a, b)
    return _wrap(a.basis_l, b.basis_r, a.operator * b.operator)
end

function *(a::SciMLLazyOperator, b::QOB.LazyOperator)
    return a * QOB.sciml_lazy_operator(b)
end

function *(a::QOB.LazyOperator, b::SciMLLazyOperator)
    return QOB.sciml_lazy_operator(a) * b
end

function *(a::SciMLLazyOperator{B1,B2}, b::QOB.Operator{B2,B3,T}) where {B1,B2,B3,T}
    return a * QOB.sciml_lazy_operator(b)
end

function *(a::QOB.Operator{B1,B2,T}, b::SciMLLazyOperator{B2,B3}) where {B1,B2,B3,T}
    return QOB.sciml_lazy_operator(a) * b
end

function *(a::SciMLLazyOperator, b::QOB.AbstractOperator)
    return a * QOB.sciml_lazy_operator(b)
end

function *(a::QOB.AbstractOperator, b::SciMLLazyOperator)
    return QOB.sciml_lazy_operator(a) * b
end

*(a::Number, b::SciMLLazyOperator) = _wrap(b.basis_l, b.basis_r, a * b.operator)
*(a::SciMLLazyOperator, b::Number) = _wrap(a.basis_l, a.basis_r, a.operator * b)
/(a::SciMLLazyOperator, b::Number) = _wrap(a.basis_l, a.basis_r, a.operator / b)

function QOB.dagger(op::SciMLLazyOperator)
    return _wrap(op.basis_r, op.basis_l, adjoint(op.operator))
end

function transpose(op::SciMLLazyOperator)
    return _wrap(op.basis_r, op.basis_l, transpose(op.operator))
end

function _operator_for_mul(op, data)
    SciMLOperators.iscached(op) && return op
    return SciMLOperators.cache_operator(op, data)
end

function mul!(result::Ket{B1}, op::SciMLLazyOperator{B1,B2}, state::Ket{B2}, alpha, beta) where {B1,B2}
    sciml_op = _operator_for_mul(op.operator, state.data)
    if isone(alpha) && iszero(beta)
        mul!(result.data, sciml_op, state.data)
        return result
    end
    mul!(result.data, sciml_op, state.data, alpha, beta)
    return result
end

function mul!(result::Bra{B2}, state::Bra{B1}, op::SciMLLazyOperator{B1,B2}, alpha, beta) where {B1,B2}
    sciml_op = _operator_for_mul(transpose(op.operator), state.data)
    if isone(alpha) && iszero(beta)
        mul!(result.data, sciml_op, state.data)
        return result
    end
    mul!(result.data, sciml_op, state.data, alpha, beta)
    return result
end

end # module
