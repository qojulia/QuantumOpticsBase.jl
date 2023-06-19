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
mutable struct LazyTensor{BL,BR,F,I,T} <: LazyOperator{BL,BR}
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

const single_dataoperator{B1,B2} = LazyTensor{B1,B2,F,I,Tuple{T}} where {B1,B2,F,I,T<:DataOperator} 
function +(a::T1,b::T2) where {T1 <: single_dataoperator{B1,B2},T2 <: single_dataoperator{B1,B2}} where {B1,B2}
    if a.indices == b.indices
        op = a.operators[1] * a.factor + b.operators[1] * b.factor
        return LazyTensor(a.basis_l, a.basis_r, a.indices, (op,))
    end
    LazySum(a) + LazySum(b)
end
function -(a::T1,b::T2) where {T1 <: single_dataoperator{B1,B2},T2 <: single_dataoperator{B1,B2}} where {B1,B2}
    if a.indices == b.indices
        op = a.operators[1] * a.factor - b.operators[1] * b.factor
        return LazyTensor(a.basis_l, a.basis_r, a.indices, (op,))
    end
    LazySum(a) - LazySum(b)
end

function tensor(a::LazyTensor{B1,B2},b::AbstractOperator{B3,B4}) where {B1,B2,B3,B4}
    if B3 <: CompositeBasis || B4 <: CompositeBasis
        throw(ArgumentError("tensor(a::LazyTensor{B1,B2},b::AbstractOperator{B3,B4}) is not implemented for B3 or B4 being CompositeBasis unless b is identityoperator "))
    else
        a ⊗ LazyTensor(b.basis_l,b.basis_r,[1],(b,),1)
    end
end
function tensor(a::AbstractOperator{B1,B2},b::LazyTensor{B3,B4})  where {B1,B2,B3,B4}
    if B1 <: CompositeBasis || B2 <: CompositeBasis
        throw(ArgumentError("tensor(a::AbstractOperator{B1,B2},b::LazyTensor{B3,B4}) is not implemented for B1 or B2 being CompositeBasis unless b is identityoperator "))
    else
        LazyTensor(a.basis_l,a.basis_r,[1],(a,),1) ⊗ b
    end
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

*(a::LazyTensor{B1,B2}, b::LazyProduct{B2,B3}) where {B1,B2,B3} = LazyProduct(a) * b
*(a::LazyProduct{B1,B2}, b::LazyTensor{B2,B3}) where {B1,B2,B3} = a * LazyProduct(b)

#*(a::LazyTensor{B1,B2}, b::AbstractOperator{B2,B3}) where {B1,B2,B3} = LazyProduct((a, b), 1)
#*(a::AbstractOperator{B1,B2}, b::LazyTensor{B2,B3}) where {B1,B2,B3} = LazyProduct((a, b), 1)
#*(a::LazyTensor{B1,B2}, b::Operator{B2,B3}) where {B1,B2,B3} = LazyProduct((a, b), 1)
#*(a::Operator{B1,B2}, b::LazyTensor{B2,B3}) where {B1,B2,B3} = LazyProduct((a, b), 1)


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

function _tp_matmul_first!(result, a::AbstractMatrix, b, α::Number, β::Number)
    d_first = size(a, 2)
    d_rest = length(b)÷d_first
    bp = parent(b)
    rp = parent(result)
    @uviews bp rp begin  # avoid allocations on reshape
        br = reshape(bp, (d_first, d_rest))
        result_r = reshape(rp, (size(a, 1), d_rest))
        mul!(result_r, a, br, α, β)
    end
    result
end

function _tp_matmul_last!(result, a::AbstractMatrix, b, α::Number, β::Number)
    d_last = size(a, 2)
    d_rest = length(b)÷d_last
    bp = parent(b)
    rp = parent(result)
    @uviews a bp rp begin  # avoid allocations on reshape
        br = reshape(bp, (d_rest, d_last))
        result_r = reshape(rp, (d_rest, size(a, 1)))
        mul!(result_r, br, transpose(a), α, β)
    end
    result
end

function _tp_matmul_get_tmp(::Type{T}, shp::NTuple{N,Int}, sym, ::Array) where {T,N}
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

function _tp_matmul_get_tmp(::Type{T}, shp::NTuple{N,Int}, sym, arr::AbstractArray) where {T,N}
    if parent(arr) === arr
        # This is a fallback that does not use the cache. Does not get triggered for arr <: Array.
        return similar(arr, T, shp)
    end
    # Unpack wrapped arrays. If we hit an Array, we will use the cache.
    # If we hit a different non-wrapped array-like, we will call `similar()`.
    _tp_matmul_get_tmp(T, shp, sym, parent(arr))
end

# reshapes of plain arrays are fine for strided, but wrappers like `Adjoint`
# can break things
_probably_strided(x::Base.ReshapedArray) = _probably_strided(parent(x))
_probably_strided(x::Array) = true
_probably_strided(x) = false

function _tp_matmul_mid!(result, a::AbstractMatrix, loc::Integer, b, α::Number, β::Number)
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

    if a isa FillArrays.Eye
        # Square Eyes are skipped higher up. This handles the non-square case. 
        size(b, loc) == size(a, 2) && size(result, loc) == size(a, 1) || throw(DimensionMismatch("Dimensions of Eye matrix do not match subspace dimensions."))
        d = min(size(a)...)

        if iszero(β)
            # Need to handle this separately, as the values in `result` may not be valid numbers
            fill!(result, zero(eltype(result)))
            if _probably_strided(result) && _probably_strided(b)
                @strided result_r[:, 1:d, :] .= α .* br[:, 1:d, :]
            else
                result_r[:, 1:d, :] .= α .* br[:, 1:d, :]
            end
        else
            rmul!(result, β)
            if _probably_strided(result) && _probably_strided(b)
                @strided result_r[:, 1:d, :] .+= α .* br[:, 1:d, :]
            else
                result_r[:, 1:d, :] .+= α .* br[:, 1:d, :]
            end
        end

        return result
    end

    # Try to "minimize" the transpose for efficiency.
    move_left = sz_b_1 < sz_b_3
    perm = move_left ? (2,1,3) : (1,3,2)

    br_p = _tp_matmul_get_tmp(eltype(br), ((size(br, i) for i in perm)...,), :_tp_matmul_mid_in_2, result)
    if _probably_strided(b)
        @strided permutedims!(br_p, br, perm)
    else
        permutedims!(br_p, br, perm)
    end

    result_r_p = _tp_matmul_get_tmp(eltype(result_r), ((size(result_r, i) for i in perm)...,), :_tp_matmul_mid_out, result)
    if _probably_strided(result)
        iszero(β) || @strided permutedims!(result_r_p, result_r, perm)
    else
        iszero(β) || permutedims!(result_r_p, result_r, perm)
    end

    if move_left
        _tp_matmul_first!(result_r_p, a, br_p, α, β)
    else
        _tp_matmul_last!(result_r_p, a, br_p, α, β)
    end

    if _probably_strided(result)
        @strided permutedims!(result_r, result_r_p, perm)
    else
        permutedims!(result_r, result_r_p, perm)
    end

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

function _tp_sum_get_tmp(op::AbstractMatrix{T}, loc::Integer, arr::AbstractArray{S,N}, arr_typeref::AbstractArray, sym) where {T,S,N}
    shp = ntuple(i -> i == loc ? size(op,1) : size(arr,i), N)
    _tp_matmul_get_tmp(promote_type(T,S), shp, sym, arr_typeref)
end

# Eyes need not be identities, but square Eyes are.
_is_square_eye(::AbstractArray) = false
_is_square_eye(::FillArrays.SquareEye) = true
_is_square_eye(x::FillArrays.Eye) = size(x, 1) == size(x, 2)
_is_square_eye(x::LinearAlgebra.Adjoint) = _is_square_eye(parent(x))
_is_square_eye(x::LinearAlgebra.Transpose) = _is_square_eye(parent(x))

#Apply a tensor product of operators to a vector.
function _tp_sum_matmul!(result_data, tp_ops, iso_ops, b_data, alpha, beta)
    if iso_ops === nothing
        ops = tp_ops
    else
        ops = (tp_ops..., iso_ops...)
    end

    n_ops = length(ops)

    # TODO: Perhaps replace with a single for loop and branch inside?
    if n_ops == 0
        if iszero(beta)
            @. result_data = alpha * b_data
        else
            @. result_data = alpha * b_data + beta * result_data
        end
    elseif n_ops == 1
        # Can add directly to the output array.
        _tp_matmul!(result_data, ops[1][1], ops[1][2], b_data, alpha, beta)
    elseif n_ops == 2
        # One temporary vector needed.
        tmp = _tp_sum_get_tmp(ops[1][1], ops[1][2], b_data, result_data, :_tp_sum_matmul_tmp1)
        _tp_matmul!(tmp, ops[1][1], ops[1][2], b_data, alpha, zero(beta))

        _tp_matmul!(result_data, ops[2][1], ops[2][2], tmp, one(alpha), beta)
    else
        # At least two temporary vectors needed.
        # Symbols identify computation stages in the cache.
        sym1 = :_tp_sum_matmul_tmp1
        sym2 = :_tp_sum_matmul_tmp2

        tmp1 = _tp_sum_get_tmp(ops[1][1], ops[1][2], b_data, result_data, sym1)
        _tp_matmul!(tmp1, ops[1][1], ops[1][2], b_data, alpha, zero(beta))

        for i in 2:n_ops-1
            tmp2 = _tp_sum_get_tmp(ops[i][1], ops[i][2], tmp1, result_data, sym2)
            _tp_matmul!(tmp2, ops[i][1], ops[i][2], tmp1, one(alpha), zero(beta))
            tmp1, tmp2 = tmp2, tmp1
            sym1, sym2 = sym2, sym1
        end

        _tp_matmul!(result_data, ops[n_ops][1], ops[n_ops][2], tmp1, one(alpha), beta)
    end

    result_data
end

# Insert explicit Eye operators where left and right bases have different sizes.
function _explicit_isometries(::Type{T}, used_indices, bl::Basis, br::Basis, shift=0) where T
    shp_l = _comp_size(bl)
    shp_r = _comp_size(br)
    shp_l != shp_r || return nothing

    isos = nothing
    iso_inds = nothing
    for (i, (sl, sr)) in enumerate(zip(shp_l, shp_r))
        if (sl != sr) && !(i + shift in used_indices)
            if isos === nothing
                isos = [Eye{T}(sl, sr)]
                iso_inds = [i + shift]
            else
                push!(isos, Eye{T}(sl, sr))
                push!(iso_inds, i + shift)
            end
        end
    end
    if isos === nothing
        return nothing
    end
    res = tuple(zip(isos, iso_inds)...)
    return res
end

# To get the shape of a CompositeBasis with number of dims inferrable at compile-time
_comp_size(b::CompositeBasis) = tuple((length(b_) for b_ in b.bases)...)
_comp_size(b::Basis) = (length(b),)

_is_pure_sparse(operators) = all(o isa Union{SparseOpPureType,EyeOpType} for o in operators)

# Prepare the tensor-product operator and indices tuple
function _tpops_tuple(operators, indices; shift=0, op_transform=identity)
    length(operators) == 0 == length(indices) && return ()

    op_pairs = tuple(((op_transform(op.data), i + shift) for (op, i) in zip(operators, indices))...)

    # Filter out identities:
    # This induces a non-trivial cost only if _is_square_eye is not inferrable.
    # This happens if we have Eyes that are not SquareEyes.
    # This can happen if the user constructs LazyTensor operators including
    # explicit identityoperator(b,b).
    filtered = filter(p->!_is_square_eye(p[1]), op_pairs)
    return filtered
end

_tpops_tuple(a::LazyTensor; shift=0, op_transform=identity) = _tpops_tuple((a.operators...,), (a.indices...,); shift, op_transform)

function mul!(result::Ket{B1}, a::LazyTensor{B1,B2,F,I,T}, b::Ket{B2}, alpha, beta) where {B1<:Basis,B2<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    iszero(alpha) && (_zero_op_mul!(result.data, beta); return result)

    if length(a.operators) > 0 && _is_pure_sparse(a.operators)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    # We reshape here so that we have the proper shape information for the
    # tensor contraction later on. Using ReshapedArray vs. reshape() avoids
    # an allocation.
    b_data = Base.ReshapedArray(b.data, _comp_size(basis(b)), ())
    result_data = Base.ReshapedArray(result.data, _comp_size(basis(result)), ())

    tp_ops = _tpops_tuple(a)
    iso_ops = _explicit_isometries(eltype(a), a.indices, a.basis_l, a.basis_r)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, b_data, alpha * a.factor, beta)

    result
end

function mul!(result::Bra{B2}, a::Bra{B1}, b::LazyTensor{B1,B2,F,I,T}, alpha, beta) where {B1<:Basis,B2<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    iszero(alpha) && (_zero_op_mul!(result.data, beta); return result)

    if length(b.operators) > 0 && _is_pure_sparse(b.operators)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    a_data = Base.ReshapedArray(a.data, _comp_size(basis(a)), ())
    result_data = Base.ReshapedArray(result.data, _comp_size(basis(result)), ())

    tp_ops = _tpops_tuple(b; op_transform=transpose)
    iso_ops = _explicit_isometries(eltype(b), b.indices, b.basis_r, b.basis_l)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, a_data, alpha * b.factor, beta)

    result
end

function mul!(result::DenseOpType{B1,B3}, a::LazyTensor{B1,B2,F,I,T}, b::DenseOpType{B2,B3}, alpha, beta) where {B1<:Basis,B2<:Basis,B3<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    iszero(alpha) && (_zero_op_mul!(result.data, beta); return result)

    if length(a.operators) > 0 && _is_pure_sparse(a.operators) && (b isa DenseOpPureType)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    b_data = Base.ReshapedArray(b.data, (_comp_size(b.basis_l)..., _comp_size(b.basis_r)...), ())
    result_data = Base.ReshapedArray(result.data, (_comp_size(result.basis_l)..., _comp_size(result.basis_r)...), ())

    tp_ops = _tpops_tuple(a)
    iso_ops = _explicit_isometries(eltype(a), a.indices, a.basis_l, a.basis_r)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, b_data, alpha * a.factor, beta)

    result
end

function mul!(result::DenseOpType{B1,B3}, a::DenseOpType{B1,B2}, b::LazyTensor{B2,B3,F,I,T}, alpha, beta) where {B1<:Basis,B2<:Basis,B3<:Basis, F,I,T<:Tuple{Vararg{DataOperator}}}
    iszero(alpha) && (_zero_op_mul!(result.data, beta); return result)

    if length(b.operators) > 0 && _is_pure_sparse(b.operators) && (a isa DenseOpPureType)
        return _mul_puresparse!(result, a, b, alpha, beta)
    end

    a_data = Base.ReshapedArray(a.data, (_comp_size(a.basis_l)..., _comp_size(a.basis_r)...), ())
    result_data = Base.ReshapedArray(result.data, (_comp_size(result.basis_l)..., _comp_size(result.basis_r)...), ())

    shft = length(a.basis_l.shape)  # b must be applied to the "B2" side of a
    tp_ops = _tpops_tuple(b; shift=shft, op_transform=transpose)
    iso_ops = _explicit_isometries(eltype(b), ((i + shft for i in b.indices)...,), b.basis_r, b.basis_l, shft)
    _tp_sum_matmul!(result_data, tp_ops, iso_ops, a_data, alpha * b.factor, beta)

    result
end


# Recursively calculate result_{IK} = \\sum_J op_{IJ} h_{JK}
function _gemm_recursive_dense_lazy(i_k, N_k, K, J, val,
                        shape, strides_k, strides_j,
                        indices, h::LazyTensor,
                        op::AbstractArray, result::AbstractArray)
    if i_k > N_k
        if isa(op, AbstractVector)
            result[K] += val*op[J]
        else I=1:size(op, 1)
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
            return nothing
        elseif !isa(h_i, EyeOpType)
            throw(ArgumentError("gemm! of LazyTensor is not implemented for $(typeof(h_i))"))
        end
    end
    @inbounds for k=1:shape[i_k]
        K_ = K + strides_k[i_k]*(k-1)
        J_ = J + strides_j[i_k]*(k-1)
        _gemm_recursive_dense_lazy(i_k + 1, N_k, K_, J_, val, shape, strides_k, strides_j, indices, h, op, result)
    end
end


# Recursively calculate result_{JI} = \\sum_K h_{JK} op_{KI}
function _gemm_recursive_lazy_dense(i_k, N_k, K, J, val,
                        shape, strides_k, strides_j,
                        indices, h::LazyTensor,
                        op::AbstractArray, result::AbstractArray)
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
            return nothing
        elseif !isa(h_i, EyeOpType)  # identity operator get handled below
            throw(ArgumentError("gemm! of LazyTensor is not implemented for $(typeof(h_i))"))
        end
    end
    @inbounds for k=1:shape[i_k]
        K_ = K + strides_k[i_k]*(k-1)
        J_ = J + strides_j[i_k]*(k-1)
        _gemm_recursive_lazy_dense(i_k + 1, N_k, K_, J_, val, shape, strides_k, strides_j, indices, h, op, result)
    end
end

"""
    check_mul!_compatibility(R, A, B)
Check that `R,A,B` are dimentially compatible for `R.=A*B`. And that `R` is not aliased with either `A` nor `B`.
"""
function check_mul!_compatibility(R::AbstractVecOrMat, A, B)
    _check_mul!_aliasing_compatibility(R, A, B)
    _check_mul!_dim_compatibility(size(R), size(A), size(B))
end
function _check_mul!_dim_compatibility(sizeR::Tuple, sizeA::Tuple, sizeB::Tuple)
    # R .= A*B
    if sizeA[2] != sizeB[1]
        throw(DimensionMismatch("A and B dimensions do not match. Can't do `A*B`"))
    end
    if sizeR != (sizeA[1], Base.tail(sizeB)...) # using tail to account for vectors
        throw(DimensionMismatch("Output dimensions do not match A*B. Can't do `R.=A*B`"))
    end
end
function _check_mul!_aliasing_compatibility(R, A, B)
    if R===A || R===B
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end
end


function _gemm_puresparse(alpha, op::AbstractArray, h::LazyTensor{B1,B2,F,I,T}, beta, result::AbstractArray) where {B1,B2,F,I,T}
    if op isa AbstractVector
        # _gemm_recursive_dense_lazy will treat `op` as a `Bra`
        _check_mul!_aliasing_compatibility(result, op, h)
        _check_mul!_dim_compatibility(size(result), reverse(size(h)), size(op))
    else
        check_mul!_compatibility(result, op, h)
    end
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    N_k = length(h.basis_r.bases)
    shape, strides_j, strides_k = _get_shape_and_strides(h)
    _gemm_recursive_dense_lazy(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

function _gemm_puresparse(alpha, h::LazyTensor{B1,B2,F,I,T}, op::AbstractArray, beta, result::AbstractArray) where {B1,B2,F,I,T}
    check_mul!_compatibility(result, h, op)
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    N_k = length(h.basis_l.bases)
    shape, strides_j, strides_k = _get_shape_and_strides(h)
    _gemm_recursive_lazy_dense(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

function _get_shape_and_strides(h)
    shape_l, shape_r = _comp_size(h.basis_l), _comp_size(h.basis_r)
    shape = min.(shape_l, shape_r)
    strides_j, strides_k = _strides(shape_l), _strides(shape_r)
    return shape, strides_j, strides_k
end

_mul_puresparse!(result::DenseOpType{B1,B3},h::LazyTensor{B1,B2,F,I,T},op::DenseOpType{B2,B3},alpha,beta) where {B1,B2,B3,F,I,T} = (_gemm_puresparse(alpha, h, op.data, beta, result.data); result)
_mul_puresparse!(result::DenseOpType{B1,B3},op::DenseOpType{B1,B2},h::LazyTensor{B2,B3,F,I,T},alpha,beta) where {B1,B2,B3,F,I,T} = (_gemm_puresparse(alpha, op.data, h, beta, result.data); result)
_mul_puresparse!(result::Ket{B1},a::LazyTensor{B1,B2,F,I,T},b::Ket{B2},alpha,beta) where {B1,B2,F,I,T} = (_gemm_puresparse(alpha, a, b.data, beta, result.data); result)
_mul_puresparse!(result::Bra{B2},a::Bra{B1},b::LazyTensor{B1,B2,F,I,T},alpha,beta) where {B1,B2,F,I,T} = (_gemm_puresparse(alpha, a.data, b, beta, result.data); result)

