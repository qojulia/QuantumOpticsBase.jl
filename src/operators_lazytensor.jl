import Base: ==, *, /, +, -

"""
    LazyTensor(b1[, b2], indices, operators[, factor=1])

Lazy implementation of a tensor product of operators.

The suboperators are stored in the `operators` field. The `indices` field
specifies in which subsystem the corresponding operator lives. Additionally,
a complex factor is stored in the `factor` field which allows for fast
multiplication with numbers.
"""
mutable struct LazyTensor{BL<:CompositeBasis,BR<:CompositeBasis,T<:Tuple{Vararg{AbstractOperator}}} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factor::ComplexF64
    indices::Vector{Int}
    operators::T

    function LazyTensor{BL,BR,T}(op::LazyTensor{BL,BR,T}, factor::Number) where {BL<:CompositeBasis,BR<:CompositeBasis,T<:Tuple{Vararg{AbstractOperator}}}
        new(op.basis_l, op.basis_r, factor, op.indices, op.operators)
    end

    function LazyTensor(basis_l::Basis, basis_r::Basis,
                        indices::Vector{Int}, ops::Vector,
                        factor::Number=1)
        if !isa(basis_l, CompositeBasis)
            basis_l = CompositeBasis(basis_l.shape, Basis[basis_l])
        end
        if !isa(basis_r, CompositeBasis)
            basis_r = CompositeBasis(basis_r.shape, Basis[basis_r])
        end
        N = length(basis_l.bases)
        @assert N==length(basis_r.bases)
        check_indices(N, indices)
        @assert length(indices) == length(ops)
        for n=1:length(indices)
            @assert isa(ops[n], AbstractOperator)
            @assert ops[n].basis_l == basis_l.bases[indices[n]]
            @assert ops[n].basis_r == basis_r.bases[indices[n]]
        end
        if !issorted(indices)
            perm = sortperm(indices)
            indices = indices[perm]
            ops = ops[perm]
        end
        ops_tup = (ops...,)
        new{typeof(basis_l),typeof(basis_r),typeof(ops_tup)}(basis_l, basis_r, complex(factor), indices, ops_tup)
    end
end
LazyTensor(op::LazyTensor{BL,BR,T}, factor::Number) where {BL<:CompositeBasis,BR<:CompositeBasis,T<:Tuple{Vararg{AbstractOperator}}} = LazyTensor{BL,BR,T}(op::LazyTensor{BL,BR,T}, factor::Number)
LazyTensor(basis::Basis, indices::Vector{Int}, ops::Vector, factor::Number=1) = LazyTensor(basis, basis, indices, ops, factor)
LazyTensor(basis_l::Basis, basis_r::Basis, index::Int, operator::AbstractOperator, factor::Number=1) = LazyTensor(basis_l, basis_r, [index], AbstractOperator[operator], factor)
LazyTensor(basis::Basis, index::Int, operators::AbstractOperator, factor::Number=1.) = LazyTensor(basis, basis, index, operators, factor)

Base.copy(x::LazyTensor) = LazyTensor(x.basis_l, x.basis_r, copy(x.indices), [copy(op) for op in x.operators], x.factor)

"""
    suboperator(op::LazyTensor, index)

Return the suboperator corresponding to the subsystem specified by `index`. Fails
if there is no corresponding operator (i.e. it would be an identity operater).
"""
suboperator(op::LazyTensor, index::Int) = op.operators[findfirst(isequal(index), op.indices)]
"""
    suboperators(op::LazyTensor, index)

Return the suboperators corresponding to the subsystems specified by `indices`. Fails
if there is no corresponding operator (i.e. it would be an identity operater).
"""
suboperators(op::LazyTensor, indices::Vector{Int}) = [op.operators[[findfirst(isequal(i), op.indices) for i in indices]]...]

dense(op::LazyTensor) = op.factor*embed(op.basis_l, op.basis_r, op.indices, DenseOperator[dense(x) for x in op.operators])
SparseArrays.sparse(op::LazyTensor) = op.factor*embed(op.basis_l, op.basis_r, op.indices, SparseOperator[sparse(x) for x in op.operators])

==(x::LazyTensor, y::LazyTensor) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factor==y.factor


# Arithmetic operations
-(a::LazyTensor) = LazyTensor(a, -a.factor)

function *(a::LazyTensor{B1,B2}, b::LazyTensor{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    indices = sort(union(a.indices, b.indices))
    ops = Vector{AbstractOperator}(undef, length(indices))
    for n in 1:length(indices)
        i = indices[n]
        in_a = i in a.indices
        in_b = i in b.indices
        if in_a && in_b
            ops[n] = suboperator(a, i)*suboperator(b, i)
        elseif in_a
            a_i = suboperator(a, i)
            ops[n] = a_i*identityoperator(typeof(a_i), b.basis_l.bases[i], b.basis_r.bases[i])
        elseif in_b
            b_i = suboperator(b, i)
            ops[n] = identityoperator(typeof(b_i), a.basis_l.bases[i], a.basis_r.bases[i])*b_i
        end
    end
    return LazyTensor(a.basis_l, b.basis_r, indices, ops, a.factor*b.factor)
end
*(a::LazyTensor, b::Number) = LazyTensor(a, a.factor*b)
*(a::Number, b::LazyTensor) = LazyTensor(b, a*b.factor)
function *(a::LazyTensor{B1,B2}, b::DenseOperator{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    result = DenseOperator{B1,B3}(a.basis_l, b.basis_r)
    gemm!(complex(1.), a, b, complex(1.), result)
    result
end
function *(a::DenseOperator{B1,B2}, b::LazyTensor{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    result = DenseOperator{B1,B3}(a.basis_l, b.basis_r)
    gemm!(complex(1.), a, b, complex(1.), result)
    result
end

/(a::LazyTensor, b::Number) = LazyTensor(a, a.factor/b)


dagger(op::LazyTensor) = LazyTensor(op.basis_r, op.basis_l, op.indices, AbstractOperator[dagger(x) for x in op.operators], conj(op.factor))

tensor(a::LazyTensor, b::LazyTensor) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, [a.indices; b.indices .+ length(a.basis_l.bases)], [a.operators..., b.operators...], a.factor*b.factor)

function tr(op::LazyTensor)
    b = basis(op)
    result = op.factor
    for i in 1:length(b.bases)
        if i in op.indices
            result *= tr(suboperator(op, i))
        else
            result *= length(b.bases[i])
        end
    end
    result
end

function ptrace(op::LazyTensor, indices::Vector{Int})
    check_ptrace_arguments(op, indices)
    N = length(op.basis_l.shape)
    rank = N - length(indices)
    factor = op.factor
    for i in indices
        if i in op.indices
            factor *= tr(suboperator(op, i))
        else
            factor *= length(op.basis_l.bases[i])
        end
    end
    remaining_indices = remove(op.indices, indices)
    if rank==1 && length(remaining_indices)==1
        return factor * suboperator(op, remaining_indices[1])
    end
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    if rank==1
        return factor * identityoperator(b_l, b_r)
    end
    ops = Vector{AbstractOperator}(undef, length(remaining_indices))
    for i in 1:length(ops)
        ops[i] = suboperator(op, remaining_indices[i])
    end
    LazyTensor(b_l, b_r, shiftremove(op.indices, indices), ops, factor)
end

normalize!(op::LazyTensor) = (op.factor /= tr(op); op)

function permutesystems(op::LazyTensor, perm::Vector{Int})
    b_l = permutesystems(op.basis_l, perm)
    b_r = permutesystems(op.basis_r, perm)
    indices = [findfirst(isequal(i), perm) for i in op.indices]
    perm_ = sortperm(indices)
    LazyTensor(b_l, b_r, indices[perm_], [op.operators[perm_]...], op.factor)
end

identityoperator(::Type{LazyTensor}, b1::Basis, b2::Basis) = LazyTensor(b1, b2, Int[], AbstractOperator[])


# Recursively calculate result_{IK} = \\sum_J op_{IJ} h_{JK}
function _gemm_recursive_dense_lazy(i_k::Int, N_k::Int, K::Int, J::Int, val::ComplexF64,
                        shape::Vector{Int}, strides_k::Vector{Int}, strides_j::Vector{Int},
                        indices::Vector{Int}, h::LazyTensor,
                        op::Matrix{ComplexF64}, result::Matrix{ComplexF64})
    if i_k > N_k
        for I=1:size(op, 1)
            result[I, K] += val*op[I, J]
        end
        return nothing
    end
    if i_k in indices
        h_i = suboperator(h, i_k)
        if isa(h_i, SparseOperator)
            h_i_data = h_i.data::SparseMatrixCSC{ComplexF64,Int}
            @inbounds for k=1:h_i_data.n
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for jptr=h_i_data.colptr[k]:h_i_data.colptr[k+1]-1
                    j = h_i_data.rowval[jptr]
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data.nzval[jptr]
                    _gemm_recursive_dense_lazy(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        elseif isa(h_i, DenseOperator)
            h_i_data = h_i.data::Matrix{ComplexF64}
            Nk = size(h_i_data, 2)
            Nj = size(h_i_data, 1)
            @inbounds for k=1:Nk
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for j=1:Nj
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data[j,k]
                    _gemm_recursive_dense_lazy(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        else
            throw(ArgumentError("gemm! of LazyTensor is not implemented for $(typeof(h_i))"))
        end
    else
        @inbounds for k=1:shape[i_k]
            K_ = K + strides_k[i_k]*(k-1)
            J_ = J + strides_j[i_k]*(k-1)
            _gemm_recursive_dense_lazy(i_k + 1, N_k, K_, J_, val, shape, strides_k, strides_j, indices, h, op, result)
        end
    end
end


# Recursively calculate result_{JI} = \\sum_K h_{JK} op_{KI}
function _gemm_recursive_lazy_dense(i_k::Int, N_k::Int, K::Int, J::Int, val::ComplexF64,
                        shape::Vector{Int}, strides_k::Vector{Int}, strides_j::Vector{Int},
                        indices::Vector{Int}, h::LazyTensor,
                        op::Matrix{ComplexF64}, result::Matrix{ComplexF64})
    if i_k > N_k
        for I=1:size(op, 2)
            result[J, I] += val*op[K, I]
        end
        return nothing
    end
    if i_k in indices
        h_i = suboperator(h, i_k)
        if isa(h_i, SparseOperator)
            h_i_data = h_i.data::SparseMatrixCSC{ComplexF64,Int}
            @inbounds for k=1:h_i_data.n
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for jptr=h_i_data.colptr[k]:h_i_data.colptr[k+1]-1
                    j = h_i_data.rowval[jptr]
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data.nzval[jptr]
                    _gemm_recursive_lazy_dense(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        elseif isa(h_i, DenseOperator)
            h_i_data = h_i.data::Matrix{ComplexF64}
            Nk = size(h_i_data, 2)
            Nj = size(h_i_data, 1)
            @inbounds for k=1:Nk
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for j=1:Nj
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data[j,k]
                    _gemm_recursive_lazy_dense(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        else
            throw(ArgumentError("gemm! of LazyTensor is not implemented for $(typeof(h_i))"))
        end
    else
        @inbounds for k=1:shape[i_k]
            K_ = K + strides_k[i_k]*(k-1)
            J_ = J + strides_j[i_k]*(k-1)
            _gemm_recursive_lazy_dense(i_k + 1, N_k, K_, J_, val, shape, strides_k, strides_j, indices, h, op, result)
        end
    end
end

function gemm(alpha::ComplexF64, op::Matrix{ComplexF64}, h::LazyTensor, beta::ComplexF64, result::Matrix{ComplexF64})
    if beta == ComplexF64(0.)
        fill!(result, beta)
    elseif beta != ComplexF64(1.)
        rmul!(result, beta)
    end
    N_k = length(h.basis_r.bases)
    shape = [min(h.basis_l.shape[i], h.basis_r.shape[i]) for i=1:length(h.basis_l.shape)]
    strides_j = _strides(h.basis_l.shape)
    strides_k = _strides(h.basis_r.shape)
    _gemm_recursive_dense_lazy(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

function gemm(alpha::ComplexF64, h::LazyTensor, op::Matrix{ComplexF64}, beta::ComplexF64, result::Matrix{ComplexF64})
    if beta == ComplexF64(0.)
        fill!(result, beta)
    elseif beta != ComplexF64(1.)
        rmul!(result, beta)
    end
    N_k = length(h.basis_l.bases)
    shape = [min(h.basis_l.shape[i], h.basis_r.shape[i]) for i=1:length(h.basis_l.shape)]
    strides_j = _strides(h.basis_l.shape)
    strides_k = _strides(h.basis_r.shape)
    _gemm_recursive_lazy_dense(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

gemm!(alpha, h::LazyTensor{B1,B2}, op::DenseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm(convert(ComplexF64, alpha), h, op.data, convert(ComplexF64, beta), result.data)
gemm!(alpha, op::DenseOperator{B1,B2}, h::LazyTensor{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm(convert(ComplexF64, alpha), op.data, h, convert(ComplexF64, beta), result.data)

function gemv!(alpha::ComplexF64, a::LazyTensor{B1,B2}, b::Ket{B2}, beta::ComplexF64, result::Ket{B1}) where {B1<:Basis,B2<:Basis}
    b_data = reshape(b.data, length(b.data), 1)
    result_data = reshape(result.data, length(result.data), 1)
    gemm(alpha, a, b_data, beta, result_data)
end

function gemv!(alpha::ComplexF64, a::Bra{B1}, b::LazyTensor{B1,B2}, beta::ComplexF64, result::Bra{B2}) where {B1<:Basis,B2<:Basis}
    a_data = reshape(a.data, 1, length(a.data))
    result_data = reshape(result.data, 1, length(result.data))
    gemm(alpha, a_data, b, beta, result_data)
end

gemv!(alpha, a::LazyTensor{B1,B2}, b::Ket{B2}, beta, result::Ket{B2}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemv!(alpha, a::Bra{B1}, b::LazyTensor{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
