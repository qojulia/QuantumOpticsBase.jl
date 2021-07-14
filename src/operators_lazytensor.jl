import Base: ==, *, /, +, -

"""
    LazyTensor(b1[, b2], indices, operators[, factor=1])

Lazy implementation of a tensor product of operators.

The suboperators are stored in the `operators` field. The `indices` field
specifies in which subsystem the corresponding operator lives. Note that these
must be sorted.
Additionally, a factor is stored in the `factor` field which allows for fast
multiplication with numbers.
"""
mutable struct LazyTensor{BL,BR,F,I,T} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factor::F
    indices::I
    operators::T
    function LazyTensor(bl::BL, br::BR, indices::I, ops::T, factor::F=_default_factor(ops)) where {BL<:CompositeBasis,BR<:CompositeBasis,F,I,T<:Tuple}
        N = length(bl.bases)
        @assert N==length(br.bases)
        check_indices(N, indices)
        @assert length(indices) == length(ops)
        @assert issorted(indices)
        for n=1:length(indices)
            @assert isa(ops[n], AbstractOperator)
            @assert ops[n].basis_l == bl.bases[indices[n]]
            @assert ops[n].basis_r == br.bases[indices[n]]
        end
        F_ = promote_type(F,map(eltype, ops)...)
        factor_ = convert(F_, factor)
        new{BL,BR,F_,I,T}(bl, br, factor_, indices, ops)
    end
end
function LazyTensor(bl::CompositeBasis, br::CompositeBasis, indices, ops::Vector, factor=_default_factor(ops))
    Base.depwarn("LazyTensor(bl, br, indices, ops::Vector, factor) is deprecated, use LazyTensor(bl, br, indices, Tuple(ops), factor) instead.",
                :LazyTensor; force=true)
    return LazyTensor(bl,br,indices,Tuple(ops),factor)
end

function LazyTensor(basis_l::CompositeBasis, basis_r::Basis, indices::I, ops, factor::F=_default_factor(ops)) where {F,I}
    br = CompositeBasis(basis_r.shape, [basis_r])
    return LazyTensor(basis_l, br, indices, ops, factor)
end
function LazyTensor(basis_l::Basis, basis_r::CompositeBasis, indices::I, ops, factor::F=_default_factor(ops)) where {F,I}
    bl = CompositeBasis(basis_l.shape, [basis_l])
    return LazyTensor(bl, basis_r, indices, ops, factor)
end
function LazyTensor(basis_l::Basis, basis_r::Basis, indices::I, ops, factor::F=_default_factor(ops)) where {F,I}
    bl = CompositeBasis(basis_l.shape, [basis_l])
    br = CompositeBasis(basis_r.shape, [basis_r])
    return LazyTensor(bl, br, indices, ops, factor)
end
LazyTensor(op::T, factor) where {T<:LazyTensor} = LazyTensor(op.basis_l, op.basis_r, op.indices, op.operators, factor)
LazyTensor(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, operator::T, factor=one(eltype(operator))) where T<:AbstractOperator = LazyTensor(basis_l, basis_r, [index], (operator,), factor)
LazyTensor(basis::Basis, index, operators, factor=_default_factor(operators)) = LazyTensor(basis, basis, index, operators, factor)

Base.copy(x::LazyTensor) = LazyTensor(x.basis_l, x.basis_r, copy(x.indices), Tuple(copy(op) for op in x.operators), x.factor)
Base.eltype(x::LazyTensor) = promote_type(eltype(x.factor), eltype.(x.operators)...)

function _default_factor(ops)
    Ts = map(eltype, ops)
    T = promote_type(Ts...)
    return one(T)
end
function _default_factor(op::T) where T<:AbstractOperator
    return one(eltype(op))
end

"""
    suboperator(op::LazyTensor, index)

Return the suboperator corresponding to the subsystem specified by `index`. Fails
if there is no corresponding operator (i.e. it would be an identity operater).
"""
suboperator(op::LazyTensor, index::Integer) = op.operators[findfirst(isequal(index), op.indices)]

"""
    suboperators(op::LazyTensor, index)

Return the suboperators corresponding to the subsystems specified by `indices`. Fails
if there is no corresponding operator (i.e. it would be an identity operater).
"""
suboperators(op::LazyTensor, indices) = [op.operators[[findfirst(isequal(i), op.indices) for i in indices]]...]

DenseOperator(op::LazyTensor) = op.factor*embed(op.basis_l, op.basis_r, op.indices, DenseOpType[DenseOperator(x) for x in op.operators])
SparseArrays.sparse(op::LazyTensor) = op.factor*embed(op.basis_l, op.basis_r, op.indices, SparseOpType[SparseOperator(x) for x in op.operators])

==(x::LazyTensor, y::LazyTensor) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factor==y.factor


# Arithmetic operations
-(a::LazyTensor) = LazyTensor(a, -a.factor)

function *(a::LazyTensor{B1,B2}, b::LazyTensor{B2,B3}) where {B1,B2,B3}
    indices = sort(union(a.indices, b.indices))
    # ops = Vector{AbstractOperator}(undef, length(indices))
    function _prod(n)
        i = indices[n]
        in_a = i in a.indices
        in_b = i in b.indices
        if in_a && in_b
            return suboperator(a, i)*suboperator(b, i)
        elseif in_a
            a_i = suboperator(a, i)
            return a_i*identityoperator(typeof(a_i), b.basis_l.bases[i], b.basis_r.bases[i])
        elseif in_b
            b_i = suboperator(b, i)
            return identityoperator(typeof(b_i), a.basis_l.bases[i], a.basis_r.bases[i])*b_i
        end
    end
    ops = Tuple(_prod(n) for n ∈ 1:length(indices))
    return LazyTensor(a.basis_l, b.basis_r, indices, ops, a.factor*b.factor)
end
*(a::LazyTensor, b::Number) = LazyTensor(a, a.factor*b)
*(a::Number, b::LazyTensor) = LazyTensor(b, a*b.factor)
function *(a::LazyTensor{B1,B2}, b::DenseOpType{B2,B3}) where {B1,B2,B3}
    result = DenseOperator(a.basis_l, b.basis_r)
    mul!(result,a,b,complex(1.),complex(1.))
    result
end
function *(a::DenseOpType{B1,B2}, b::LazyTensor{B2,B3}) where {B1,B2,B3}
    result = DenseOperator(a.basis_l, b.basis_r)
    mul!(result,a,b,complex(1.),complex(1.))
    result
end

/(a::LazyTensor, b::Number) = LazyTensor(a, a.factor/b)


dagger(op::LazyTensor) = LazyTensor(op.basis_r, op.basis_l, op.indices, Tuple(dagger(x) for x in op.operators), conj(op.factor))

tensor(a::LazyTensor, b::LazyTensor) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, [a.indices; b.indices .+ length(a.basis_l.bases)], (a.operators..., b.operators...), a.factor*b.factor)

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

function ptrace(op::LazyTensor, indices)
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
    ops = Tuple(suboperator(op, idx) for idx ∈ remaining_indices)
    LazyTensor(b_l, b_r, shiftremove(op.indices, indices), ops, factor)
end

normalize!(op::LazyTensor) = (op.factor /= tr(op); op)

function permutesystems(op::LazyTensor, perm)
    b_l = permutesystems(op.basis_l, perm)
    b_r = permutesystems(op.basis_r, perm)
    indices = [findfirst(isequal(i), perm) for i in op.indices]
    perm_ = sortperm(indices)
    LazyTensor(b_l, b_r, indices[perm_], op.operators[perm_], op.factor)
end

identityoperator(::Type{LazyTensor}, ::Type{T<:Number}, b1::Basis, b2::Basis) where T<:Number = LazyTensor(b1, b2, Int[], Tuple{}(), one(T))


# Recursively calculate result_{IK} = \\sum_J op_{IJ} h_{JK}
function _gemm_recursive_dense_lazy(i_k, N_k, K, J, val,
                        shape, strides_k, strides_j,
                        indices, h::LazyTensor,
                        op::Matrix, result::Matrix)
    if i_k > N_k
        for I=1:size(op, 1)
            result[I, K] += val*op[I, J]
        end
        return nothing
    end
    if i_k in indices
        h_i = suboperator(h, i_k)
        if isa(h_i, SparseOpPureType)
            h_i_data = h_i.data
            @inbounds for k=1:h_i_data.n
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for jptr=h_i_data.colptr[k]:h_i_data.colptr[k+1]-1
                    j = h_i_data.rowval[jptr]
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data.nzval[jptr]
                    _gemm_recursive_dense_lazy(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        elseif isa(h_i, DataOperator)
            h_i_data = h_i.data
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
function _gemm_recursive_lazy_dense(i_k, N_k, K, J, val,
                        shape, strides_k, strides_j,
                        indices, h::LazyTensor,
                        op::Matrix, result::Matrix)
    if i_k > N_k
        for I=1:size(op, 2)
            result[J, I] += val*op[K, I]
        end
        return nothing
    end
    if i_k in indices
        h_i = suboperator(h, i_k)
        if isa(h_i, SparseOpPureType) # TODO: adjoint sparse matrices
            h_i_data = h_i.data
            @inbounds for k=1:h_i_data.n
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for jptr=h_i_data.colptr[k]:h_i_data.colptr[k+1]-1
                    j = h_i_data.rowval[jptr]
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data.nzval[jptr]
                    _gemm_recursive_lazy_dense(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        elseif isa(h_i, DataOperator)
            h_i_data = h_i.data
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

function gemm(alpha, op::Matrix, h::LazyTensor, beta, result::Matrix)
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    N_k = length(h.basis_r.bases)
    shape = [min(h.basis_l.shape[i], h.basis_r.shape[i]) for i=1:length(h.basis_l.shape)]
    strides_j = _strides(h.basis_l.shape)
    strides_k = _strides(h.basis_r.shape)
    _gemm_recursive_dense_lazy(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

function gemm(alpha, h::LazyTensor, op::Matrix, beta, result::Matrix)
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    N_k = length(h.basis_l.bases)
    shape = [min(h.basis_l.shape[i], h.basis_r.shape[i]) for i=1:length(h.basis_l.shape)]
    strides_j = _strides(h.basis_l.shape)
    strides_k = _strides(h.basis_r.shape)
    _gemm_recursive_lazy_dense(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

mul!(result::DenseOpType{B1,B3},h::LazyTensor{B1,B2},op::DenseOpType{B2,B3},alpha,beta) where {B1,B2,B3} = (gemm(alpha, h, op.data, beta, result.data); result)
mul!(result::DenseOpType{B1,B3},op::DenseOpType{B1,B2},h::LazyTensor{B2,B3},alpha,beta) where {B1,B2,B3} = (gemm(alpha, op.data, h, beta, result.data); result)

function mul!(result::Ket{B1},a::LazyTensor{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2}
    b_data = reshape(b.data, length(b.data), 1)
    result_data = reshape(result.data, length(result.data), 1)
    gemm(alpha, a, b_data, beta, result_data)
    result
end

function mul!(result::Bra{B2},a::Bra{B1},b::LazyTensor{B1,B2},alpha,beta) where {B1,B2}
    a_data = reshape(a.data, 1, length(a.data))
    result_data = reshape(result.data, 1, length(result.data))
    gemm(alpha, a_data, b, beta, result_data)
    result
end
