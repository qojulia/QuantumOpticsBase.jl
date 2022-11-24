import Base: isequal, ==, *, /, +, -

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
        F_ = promote_type(F, mapreduce(eltype, promote_type, ops; init=F))
        factor_ = convert(F_, factor)
        new{BL,BR,F_,I,T}(bl, br, factor_, indices, ops)
    end
end
function LazyTensor(bl::CompositeBasis, br::CompositeBasis, indices::I, ops::Vector, factor::F=_default_factor(ops)) where {F,I}
    Base.depwarn("LazyTensor(bl, br, indices, ops::Vector, factor) is deprecated, use LazyTensor(bl, br, indices, Tuple(ops), factor) instead.",
                :LazyTensor; force=true)
    return LazyTensor(bl,br,indices,Tuple(ops),factor)
end

function LazyTensor(basis_l::CompositeBasis, basis_r::Basis, indices::I, ops::T, factor::F=_default_factor(ops)) where {F,I,T<:Tuple}
    br = CompositeBasis(basis_r.shape, [basis_r])
    return LazyTensor(basis_l, br, indices, ops, factor)
end
function LazyTensor(basis_l::Basis, basis_r::CompositeBasis, indices::I, ops::T, factor::F=_default_factor(ops)) where {F,I,T<:Tuple}
    bl = CompositeBasis(basis_l.shape, [basis_l])
    return LazyTensor(bl, basis_r, indices, ops, factor)
end
function LazyTensor(basis_l::Basis, basis_r::Basis, indices::I, ops::T, factor::F=_default_factor(ops)) where {F,I,T<:Tuple}
    bl = CompositeBasis(basis_l.shape, [basis_l])
    br = CompositeBasis(basis_r.shape, [basis_r])
    return LazyTensor(bl, br, indices, ops, factor)
end
LazyTensor(op::T, factor) where {T<:LazyTensor} = LazyTensor(op.basis_l, op.basis_r, op.indices, op.operators, factor)
LazyTensor(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, operator::T, factor=one(eltype(operator))) where T<:AbstractOperator = LazyTensor(basis_l, basis_r, [index], (operator,), factor)
LazyTensor(basis::Basis, index, operators, factor=_default_factor(operators)) = LazyTensor(basis, basis, index, operators, factor)

Base.copy(x::LazyTensor) = LazyTensor(x.basis_l, x.basis_r, copy(x.indices), Tuple(copy(op) for op in x.operators), x.factor)
function Base.eltype(x::LazyTensor)
    F = eltype(x.factor)
    promote_type(F, mapreduce(eltype, promote_type, x.operators; init=F))
end

function _default_factor(ops)
    T = mapreduce(eltype, promote_type, ops)
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

isequal(x::LazyTensor, y::LazyTensor) = samebases(x,y) && isequal(x.indices, y.indices) && isequal(x.operators, y.operators) && isequal(x.factor, y.factor)
==(x::LazyTensor, y::LazyTensor) = samebases(x,y) && x.indices==y.indices && x.operators==y.operators && x.factor==y.factor


# Arithmetic operations
-(a::LazyTensor) = LazyTensor(a, -a.factor)

function +(a::LazyTensor{B1,B2}, b::LazyTensor{B1,B2}) where {B1,B2}
    if length(a.indices) == 1 && a.indices == b.indices
        op = a.operators[1] * a.factor + b.operators[1] * b.factor
        return LazyTensor(a.basis_l, a.basis_r, a.indices, (op,))
    end
    throw(ArgumentError("Addition of LazyTensor operators is only defined in case both operators act nontrivially on the same, single tensor factor."))
end

function -(a::LazyTensor{B1,B2}, b::LazyTensor{B1,B2}) where {B1,B2}
    if length(a.indices) == 1 && a.indices == b.indices
        op = a.operators[1] * a.factor - b.operators[1] * b.factor
        return LazyTensor(a.basis_l, a.basis_r, a.indices, (op,))
    end
    throw(ArgumentError("Subtraction of LazyTensor operators is only defined in case both operators act nontrivially on the same, single tensor factor."))
end

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

identityoperator(::Type{<:LazyTensor}, ::Type{T}, b1::Basis, b2::Basis) where {T<:Number} = LazyTensor(b1, b2, Int[], Tuple{}(), one(T))

## LazyTensor global cache

function lazytensor_default_cache_size()
    return Int(min(Int64(1)<<32, Sys.total_memory()>>2, typemax(Int)))
end

# NOTE: By specifying a union type here, the compiler should find it easy to
#       automatically specialize on the vector types and avoid runtime dispatch.
const LazyTensorCacheable = Union{Vector{ComplexF64}, Vector{ComplexF32}, Vector{Float64}, Vector{Float32}}
const lazytensor_cache = LRU{Tuple{Symbol, UInt, UInt, DataType}, LazyTensorCacheable}(; maxsize=lazytensor_default_cache_size(), by=sizeof)

taskid() = convert(UInt, pointer_from_objref(current_task()))
const _lazytensor_use_cache = Ref(true)
lazytensor_use_cache() = _lazytensor_use_cache[]

"""
    lazytensor_clear_cache()
Clear the current contents of the cache.
"""
function lazytensor_clear_cache()
    empty!(lazytensor_cache)
    return
end

"""
    lazytensor_cachesize()
Return the current memory size (in bytes) of all the objects in the cache.
"""
lazytensor_cachesize() = lazytensor_cache.currentsize

"""
    lazytensor_disable_cache()
Disable the cache for further use but does not clear its current contents.
Also see [`lazytensor_clear_cache()`](@ref)
"""
function lazytensor_disable_cache()
    _lazytensor_use_cache[] = false
    return
end

"""
    lazytensor_enable_cache(; maxsize::Int = ..., maxrelsize::Real = ...)
(Re)-enable the cache for further use; set the maximal size `maxsize` (as number of bytes)
or relative size `maxrelsize`, as a fraction between 0 and 1, resulting in
`maxsize = floor(Int, maxrelsize * Sys.total_memory())`. Default value is `maxsize = 2^32` bytes, which amounts to 4 gigabytes of memory.
"""
function lazytensor_enable_cache(; maxsize::Int = -1, maxrelsize::Real = 0.0)
    if maxsize == -1 && maxrelsize == 0.0
        maxsize = lazytensor_default_cache_size()
    elseif maxrelsize > 0
        maxsize = max(maxsize, floor(Int, maxrelsize*Sys.total_memory()))
    else
        @assert maxsize >= 0
    end
    _lazytensor_use_cache[] = true
    resize!(lazytensor_cache; maxsize = maxsize)
    return
end

function _tp_matmul_first!(result::Base.ReshapedArray, a::AbstractMatrix, b::Base.ReshapedArray, α::Number, β::Number)
    d_first = size(b, 1)
    d_rest = length(b)÷d_first
    bp = b.parent
    rp = result.parent
    @uviews bp rp begin  # avoid allocations on reshape
        br = reshape(bp, (d_first, d_rest))
        result_r = reshape(rp, (size(a, 1), d_rest))
        mul!(result_r, a, br, α, β)
    end
    result
end

function _tp_matmul_last!(result::Base.ReshapedArray, a::AbstractMatrix, b::Base.ReshapedArray, α::Number, β::Number)
    d_last = size(b, ndims(b))
    d_rest = length(b)÷d_last
    bp = b.parent
    rp = result.parent
    @uviews a bp rp begin  # avoid allocations on reshape
        br = reshape(bp, (d_rest, d_last))
        result_r = reshape(rp, (d_rest, size(a, 1)))
        mul!(result_r, br, transpose(a), α, β)
    end
    result
end

function _tp_matmul_get_tmp(::Type{T}, shp::NTuple{N,Int}, sym) where {T,N}
    len = prod(shp)
    use_cache = lazytensor_use_cache()
    key = (sym, taskid(), UInt(len), T)
    if use_cache && Vector{T} <: LazyTensorCacheable
        tmp::Vector{T} = get!(lazytensor_cache, key) do
            Vector{T}(undef, len)
        end
    else
        tmp = Vector{T}(undef, len)
    end
    Base.ReshapedArray(tmp, shp, ())
end

function _tp_matmul_mid!(result::Base.ReshapedArray, a::AbstractMatrix, loc::Integer, b::Base.ReshapedArray, α::Number, β::Number)
    sz_b_1 = 1
    for i in 1:loc-1
        sz_b_1 *= size(b,i)
    end
    sz_b_3 = 1
    for i in loc+1:ndims(b)
        sz_b_3 *= size(b,i)
    end

    # TODO: Perhaps we should avoid reshaping here... should be possible to infer
    # contraction index tuple sizes
    br = Base.ReshapedArray(b, (sz_b_1, size(b, loc), sz_b_3), ())
    result_r = Base.ReshapedArray(result, (sz_b_1, size(a, 1), sz_b_3), ())

    # Try to "minimize" the transpose for efficiency.
    move_left = sz_b_1 < sz_b_3
    perm = move_left ? (2,1,3) : (1,3,2)

    br_p = _tp_matmul_get_tmp(eltype(br), ((size(br, i) for i in perm)...,), :_tp_matmul_mid_in)
    @strided permutedims!(br_p, br, perm)
    #permutedims!(br_p, br, perm)

    result_r_p = _tp_matmul_get_tmp(eltype(result_r), ((size(result_r, i) for i in perm)...,), :_tp_matmul_mid_out)
    β == 0.0 || @strided permutedims!(result_r_p, result_r, perm)
    #β == 0.0 || permutedims!(result_r_p, result_r, perm)

    if move_left
        _tp_matmul_first!(result_r_p, a, br_p, α, β)
    else
        _tp_matmul_last!(result_r_p, a, br_p, α, β)
    end

    @strided permutedims!(result_r, result_r_p, perm)
    #permutedims!(result_r, result_r_p, perm)

    result
end

function _tp_matmul!(result, a::AbstractMatrix, loc::Integer, b, α::Number, β::Number)
    # Apply a matrix `α * a` to one tensor factor of a tensor `b`.
    # If β is nonzero, add to β times `result`. In other words, we do:
    # result = α * a * b + β * result
    #
    # Parameters:
    #     result: Array to hold the output tensor.
    #     a: Matrix to apply.
    #     loc: Index of the dimension of `b` to which `a` should be applied.
    #     b: Array containing the tensor to which `a` should be applied.
    #     α: Factor multiplying `a`.
    #     β: Factor multiplying `result`.
    # Returns:
    #     The modified `result`.
    #
    # Use GEMM directly where possible, otherwise we have to permute
    if loc == 1
        return _tp_matmul_first!(result, a, b, α, β)
    elseif loc == ndims(b)
        return _tp_matmul_last!(result, a, b, α, β)
    end
    _tp_matmul_mid!(result, a, loc, b, α, β)
end

function _tp_sum_get_tmp(op::AbstractMatrix{T}, loc::Integer, arr::AbstractArray{S,N}, sym) where {T,S,N}
    shp = ntuple(i -> i == loc ? size(op,1) : size(arr,i), N)
    _tp_matmul_get_tmp(promote_type(T,S), shp, sym)
end

#Apply a tensor product of operators to a vector.
function _tp_sum_matmul!(result_data, tp_ops, iso_ops, b_data, alpha, beta)
    iszero(alpha) && return rmul!(result_data, beta)

    if iso_ops === nothing
        n_ops = length(tp_ops[1])
        ops = zip(tp_ops...)
    else
        n_ops = length(tp_ops[1]) + length(iso_ops[1])
        ops = Iterators.flatten((zip(tp_ops...), zip(iso_ops...)))
    end

    # TODO: Perhaps replace with a single for loop and branch inside?
    if n_ops == 0
        result_data .= alpha .* b_data + beta .* result_data
    elseif n_ops == 1
        # Can add directly to the output array.
        _tp_matmul!(result_data, first(ops)..., b_data, alpha, beta)
    elseif n_ops == 2
        # One temporary vector needed.
        op1, istate = iterate(ops)::Tuple # "not-nothing" assertion to help type inference
        tmp = _tp_sum_get_tmp(op1..., b_data, :_tp_sum_matmul_tmp1)
        _tp_matmul!(tmp, op1..., b_data, alpha, zero(beta))

        op2, istate = iterate(ops, istate)::Tuple # "not-nothing" assertion to help type inference
        _tp_matmul!(result_data, op2..., tmp, one(alpha), beta)
    else
        # At least two temporary vectors needed.
        # Symbols identify computation stages in the cache.
        sym1 = :_tp_sum_matmul_tmp1
        sym2 = :_tp_sum_matmul_tmp2

        op1, istate = iterate(ops)::Tuple # "not-nothing" assertion to help type inference
        tmp1 = _tp_sum_get_tmp(op1..., b_data, sym1)
        _tp_matmul!(tmp1, op1..., b_data, alpha, zero(beta))

        next = iterate(ops, istate)::Tuple # "not-nothing" assertion to help type inference
        for _ in 2:n_ops-1
            op, istate = next
            tmp2 = _tp_sum_get_tmp(op..., tmp1, sym2)
            _tp_matmul!(tmp2, op..., tmp1, one(alpha), zero(beta))
            tmp1, tmp2 = tmp2, tmp1
            sym1, sym2 = sym2, sym1
            next = iterate(ops, istate)::Tuple # "not-nothing" assertion to help type inference
        end

        op, istate = next
        _tp_matmul!(result_data, op..., tmp1, one(alpha), beta)
    end

    result_data
end

# Represents a rectangular "identity" matrix of ones along the diagonal.
struct _SimpleIsometry{I<:Integer}  # We have no need to subtype AbstractMatrix
    shape::Tuple{I,I}
    function _SimpleIsometry(d1::I, d2::I) where {I<:Integer}
        new{I}((d1, d2))
    end
end
Base.size(A::_SimpleIsometry) = A.shape
Base.size(A::_SimpleIsometry, i) = A.shape[i]

function _tp_sum_get_tmp(op::_SimpleIsometry, loc::Integer, arr::AbstractArray{S,N}, sym) where {S,N}
    shp = ntuple(i -> i == loc ? size(op,1) : size(arr,i), N)
    _tp_matmul_get_tmp(S, shp, sym)
end

function _tp_matmul!(result, a::_SimpleIsometry, loc::Integer, b, α::Number, β::Number)
    shp_b_1 = 1
    for i in 1:loc-1
        shp_b_1 *= size(b,i)
    end
    shp_b_3 = 1
    for i in loc+1:ndims(b)
        shp_b_3 *= size(b,i)
    end

    @assert size(b, loc) == size(a, 2)

    br = Base.ReshapedArray(b, (shp_b_1, size(b, loc), shp_b_3), ())
    result_r = Base.ReshapedArray(result, (shp_b_1, size(a, 1), shp_b_3), ())

    d = min(size(a)...)

    if β != 0
        rmul!(result, β)
        @strided result_r[:, 1:d, :] .+= α .* br[:, 1:d, :]
    else
        @strided result_r[:, 1:d, :] .= α .* br[:, 1:d, :]
        @strided result_r[:, d+1:end, :] .= zero(eltype(result))
    end

    result
end

function _explicit_isometries(used_indices, bl::Basis, br::Basis, shift=0)
    shp_l = _comp_size(bl)
    shp_r = _comp_size(br)
    shp_l != shp_r || return nothing

    isos = nothing
    iso_inds = nothing
    for (i, (sl, sr)) in enumerate(zip(shp_l, shp_r))
        if (sl != sr) && !(i + shift in used_indices)
            if isos === nothing
                isos = [_SimpleIsometry(sl, sr)]
                iso_inds = [i + shift]
            else
                push!(isos, _SimpleIsometry(sl, sr))
                push!(iso_inds, i + shift)
            end
        end
    end
    if isos === nothing
        return nothing
    end
    isos, iso_inds
end

# To get the shape of a CompositeBasis with number of dims inferrable at compile-time
_comp_size(b::CompositeBasis) = tuple((length(b_) for b_ in b.bases)...)
_comp_size(b::Basis) = (length(b),)

function mul!(result::Ket{B1}, a::LazyTensor{B1,B2,F,I,T}, b::Ket{B2}, alpha, beta) where {B1<:Basis,B2<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    if length(a.operators) > 0 && all(o isa SparseOpPureType for o in a.operators)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    # We reshape here so that we have the proper shape information for the
    # tensor contraction later on. Using ReshapedArray vs. reshape() avoids
    # an allocation.
    b_data = Base.ReshapedArray(b.data, _comp_size(basis(b)), ())
    result_data = Base.ReshapedArray(result.data, _comp_size(basis(result)), ())

    tp_ops = (tuple((op.data for op in a.operators)...), a.indices)
    iso_ops = _explicit_isometries(a.indices, a.basis_l, a.basis_r)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, b_data, alpha * a.factor, beta)

    result
end

function mul!(result::Bra{B2}, a::Bra{B1}, b::LazyTensor{B1,B2,F,I,T}, alpha, beta) where {B1<:Basis,B2<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    if length(b.operators) > 0 && all(o isa SparseOpPureType for o in b.operators)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    a_data = Base.ReshapedArray(a.data, _comp_size(basis(a)), ())
    result_data = Base.ReshapedArray(result.data, _comp_size(basis(result)), ())

    tp_ops = (tuple((transpose(op.data) for op in b.operators)...), b.indices)
    iso_ops = _explicit_isometries(b.indices, b.basis_r, b.basis_l)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, a_data, alpha * b.factor, beta)

    result
end

function mul!(result::DenseOpType{B1,B3}, a::LazyTensor{B1,B2,F,I,T}, b::DenseOpType{B2,B3}, alpha, beta) where {B1<:Basis,B2<:Basis,B3<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    if length(a.operators) > 0 && all(o isa SparseOpPureType for o in a.operators) && (b isa DenseOpPureType)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    b_data = Base.ReshapedArray(b.data, (_comp_size(b.basis_l)..., _comp_size(b.basis_r)...), ())
    result_data = Base.ReshapedArray(result.data, (_comp_size(result.basis_l)..., _comp_size(result.basis_r)...), ())

    tp_ops = (tuple((op.data for op in a.operators)...), a.indices)
    iso_ops = _explicit_isometries(a.indices, a.basis_l, a.basis_r)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, b_data, alpha * a.factor, beta)

    result
end

function mul!(result::DenseOpType{B1,B3}, a::DenseOpType{B1,B2}, b::LazyTensor{B2,B3,F,I,T}, alpha, beta) where {B1<:Basis,B2<:Basis,B3<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    if length(b.operators) > 0 && all(o isa SparseOpPureType for o in b.operators) && (a isa DenseOpPureType)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    a_data = Base.ReshapedArray(a.data, (_comp_size(a.basis_l)..., _comp_size(a.basis_r)...), ())
    result_data = Base.ReshapedArray(result.data, (_comp_size(result.basis_l)..., _comp_size(result.basis_r)...), ())

    shft = length(a.basis_l.shape)  # b must be applied to the "B2" side of a
    tp_ops = (tuple((transpose(op.data) for op in b.operators)...), tuple((i + shft for i in b.indices)...))
    iso_ops = _explicit_isometries(tp_ops[2], b.basis_r, b.basis_l, shft)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, a_data, alpha * b.factor, beta)

    result
end


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

function _gemm_puresparse(alpha, op::Matrix, h::LazyTensor{B1,B2,F,I,T}, beta, result::Matrix) where {B1,B2,F,I,T<:Tuple{Vararg{SparseOpPureType}}}
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

function _gemm_puresparse(alpha, h::LazyTensor{B1,B2,F,I,T}, op::Matrix, beta, result::Matrix) where {B1,B2,F,I,T<:Tuple{Vararg{SparseOpPureType}}}
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

_mul_puresparse!(result::DenseOpType{B1,B3},h::LazyTensor{B1,B2,F,I,T},op::DenseOpType{B2,B3},alpha,beta) where {B1,B2,B3,F,I,T<:Tuple{Vararg{SparseOpPureType}}} = (_gemm_puresparse(alpha, h, op.data, beta, result.data); result)
_mul_puresparse!(result::DenseOpType{B1,B3},op::DenseOpType{B1,B2},h::LazyTensor{B2,B3,F,I,T},alpha,beta) where {B1,B2,B3,F,I,T<:Tuple{Vararg{SparseOpPureType}}} = (_gemm_puresparse(alpha, op.data, h, beta, result.data); result)

function _mul_puresparse!(result::Ket{B1},a::LazyTensor{B1,B2,F,I,T},b::Ket{B2},alpha,beta) where {B1,B2,F,I,T<:Tuple{Vararg{SparseOpPureType}}}
    b_data = reshape(b.data, length(b.data), 1)
    result_data = reshape(result.data, length(result.data), 1)
    _gemm_puresparse(alpha, a, b_data, beta, result_data)
    result
end

function _mul_puresparse!(result::Bra{B2},a::Bra{B1},b::LazyTensor{B1,B2,F,I,T},alpha,beta) where {B1,B2,F,I,T<:Tuple{Vararg{SparseOpPureType}}}
    a_data = reshape(a.data, 1, length(a.data))
    result_data = reshape(result.data, 1, length(result.data))
    _gemm_puresparse(alpha, a_data, b, beta, result_data)
    result
end
