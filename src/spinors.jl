"""
    SumBasis(b1, b2...)

Similar to [`CompositeBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumBasis{S,B} <: Basis
    shape::S
    bases::B
end
SumBasis(bases) = SumBasis(Int[length(b) for b in bases], bases)
SumBasis(shape, bases::Vector) = (tmp = (bases...,); SumBasis(shape, tmp))
SumBasis(bases::Vector) = SumBasis((bases...,))
SumBasis(bases::Basis...) = SumBasis((bases...,))

==(b1::T, b2::T) where T<:SumBasis = equal_shape(b1.shape, b2.shape)
==(b1::SumBasis, b2::SumBasis) = false
length(b::SumBasis) = sum(b.shape)

"""
    directsum(b1::Basis, b2::Basis)

Construct the [`SumBasis`](@ref) out of two sub-bases.
"""
directsum(b1::Basis, b2::Basis) = SumBasis(Int[length(b1); length(b2)], Basis[b1, b2])
directsum(b::Basis) = b
directsum(b::Basis...) = reduce(directsum, b)
function directsum(b1::SumBasis, b2::Basis)
    shape = [b1.shape;length(b2)]
    bases = [b1.bases...;b2]
    return SumBasis(shape, (bases...,))
end
function directsum(b1::Basis, b2::SumBasis)
    shape = [length(b1);b2.shape]
    bases = [b1;b2.bases...]
    return SumBasis(shape, (bases...,))
end
function directsum(b1::SumBasis, b2::SumBasis)
    shape = [b1.shape;b2.shape]
    bases = [b1.bases...;b2.bases...]
    return SumBasis(shape, (bases...,))
end
directsum() = GenericBasis(0)

"""
    directsum(x::Ket, y::Ket)

Construct a spinor via the [`directsum`](@ref) of two [`Ket`](@ref)s.
The result is a [`Ket`](@ref) with data given by `[x.data;y.data]` and its
basis given by the corresponding [`SumBasis`](@ref).
**NOTE**: The resulting state is not normalized!
"""
directsum(x::Ket, y::Ket) = Ket(directsum(x.basis, y.basis), [x.data; y.data])
directsum(x::StateVector...) = reduce(directsum, x)

"""
    getblock(x::Ket{<:SumBasis}, i)

For a [`Ket`](@ref) defined on a [`SumBasis`](@ref), get the state as it is defined
on the ith sub-basis.
"""
function getblock(x::Ket{B}, i) where B<:SumBasis
    b_i = x.basis.bases[i]
    inds = cumsum([0;length.(x.basis.bases[1:i])...])
    return Ket(b_i, x.data[inds[i]+1:inds[i+1]])
end

"""
    setblock!(x::Ket{<:SumBasis}, val::Ket, i)

Set the data of `x` on the ith sub-basis equal to the data of `val`.
"""
function setblock!(x::Ket{B}, val::Ket, i) where B<:SumBasis
    check_samebases(x.basis.bases[i], val)
    inds = cumsum([0;length.(x.basis.bases[1:i])...])
    x.data[inds[i]+1:inds[i+1]].data .= val.data
    return x
end

"""
    directsum(x::DataOperator, y::DataOperator)

Compute the direct sum of two operators. The result is an operator on the
corresponding [`SumBasis`](@ref).
"""
function directsum(a::DataOperator, b::DataOperator)
    dType = promote_type(eltype(a),eltype(b))
    data = zeros(dType, size(a,1)+size(b,1), size(a,2)+size(b,2))
    data[1:size(a,1),1:size(a,2)] = a.data
    data[size(a,1)+1:end, size(a,2)+1:end] = b.data
    return Operator(directsum(a.basis_l, b.basis_l), directsum(a.basis_r, b.basis_r), data)
end
function directsum(a::SparseOpType, b::SparseOpType)
    dType = promote_type(eltype(a),eltype(b))
    data = spzeros(dType, size(a,1)+size(b,1), size(a,2)+size(b,2))
    data[1:size(a,1),1:size(a,2)] = a.data
    data[size(a,1)+1:end, size(a,2)+1:end] = b.data
    return Operator(directsum(a.basis_l, b.basis_l), directsum(a.basis_r, b.basis_r), data)
end
directsum(a::AbstractOperator...) = reduce(directsum, a)

"""
    setblock!(op::DataOperator{<:SumBasis,<:SumBasis}, val::DataOperator, i, j)

Set the data of `op` corresponding to the block `(i,j)` equal to the data of `val`.
"""
function setblock!(op::DataOperator{<:SumBasis,<:SumBasis}, val::DataOperator, i, j)
    (bases_l,bases_r) = op.basis_l.bases, op.basis_r.bases
    check_samebases(bases_l[i], val.basis_l)
    check_samebases(bases_r[j], val.basis_r)
    inds_i = cumsum([0;length.(bases_l[1:i])...])
    inds_j = cumsum([0;length.(bases_r[1:j])...])
    op.data[inds_i[i]+1:inds_i[i+1],inds_j[j]+1:inds_j[j+1]] = val.data
    return op
end

"""
    getblock(op::Operator{<:SumBasis,<:SumBasis}, i, j)

Get the sub-basis operator corresponding to the block `(i,j)` of `op`.
"""
function getblock(op::DataOperator{BL,BR}, i, j) where {BL<:SumBasis,BR<:SumBasis}
    (bases_l,bases_r) = op.basis_l.bases, op.basis_r.bases
    inds_i = cumsum([0;length.(bases_l[1:i])...])
    inds_j = cumsum([0;length.(bases_r[1:j])...])
    data = op.data[inds_i[i]+1:inds_i[i+1],inds_j[j]+1:inds_j[j+1]]
    return Operator(bases_l[i],bases_r[j], data)
end

"""
    embed(basis_l::SumBasis, basis_r::SumBasis,
               index::Integer, operator)

Embed an operator defined on a single subspace specified by the `index` into
a [`SumBasis`](@ref).
"""
function embed(basis_l::SumBasis, basis_r::SumBasis,
               index::Integer, op::T) where T<:DataOperator
    @assert length(basis_r.bases) == length(basis_l.bases)

    basis_l.bases[index] == op.basis_l || throw(IncompatibleBases())
    basis_r.bases[index] == op.basis_r || throw(IncompatibleBases())

    embedded_op = SparseOperator(eltype(op), basis_l, basis_r)
    setblock!(embedded_op, op, index, index)
    return embedded_op
end

"""
    embed(basis_l::SumBasis, basis_r::SumBasis,
                indices, operator)

Embed an operator defined on multiple subspaces specified by the `indices` into
a [`SumBasis`](@ref).
"""
function embed(basis_l::SumBasis, basis_r::SumBasis,
               indices, op::T) where T<:DataOperator
    @assert length(basis_r.bases) == length(basis_l.bases)

    embedded_op = SparseOperator(eltype(op), basis_l, basis_r)
    for i=1:length(indices), j=1:length(indices)
        op_ = getblock(op, i, j)
        setblock!(embedded_op, op_, indices[i], indices[j])
    end
    return embedded_op
end

"""
    embed(basis_l::SumBasis, basis_r::SumBasis,
               indices, operators)

Embed a list of operators on subspaces specified by the `indices` into a
[`SumBasis`](@ref).
"""
function embed(basis_l::SumBasis, basis_r::SumBasis,
               indices, ops::Union{Tuple{Vararg{<:DataOperator}},Vector{<:DataOperator}})
    @assert length(basis_r.bases) == length(basis_l.bases)

    T = mapreduce(eltype, promote_type, ops)
    embedded_op = SparseOperator(T, basis_l, basis_r)
    for k=1:length(ops)
        op = ops[k]
        idx = indices[k]
        if length(idx)==1
            setblock!(embedded_op, op, idx[1], idx[1])
        else
            for i=1:length(idx), j=1:length(idx)
                op_ = getblock(op, i, j)
                setblock!(embedded_op, op_, idx[i], idx[j])
            end
        end
    end
    return embedded_op
end

embed(b::SumBasis, indices, ops) = embed(b, b, indices, ops)


"""
    LazyDirectSum <: AbstractOperator

Lazy implementation of `directsum`
"""
mutable struct LazyDirectSum{BL,BR,T} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    operators::T
end

# Methods
LazyDirectSum(op1::AbstractOperator, op2::AbstractOperator) = LazyDirectSum(directsum(op1.basis_l,op2.basis_l),directsum(op1.basis_r,op2.basis_r),(op1,op2))
LazyDirectSum(op1::LazyDirectSum, op2::AbstractOperator) = LazyDirectSum(directsum(op1.basis_l,op2.basis_l),directsum(op1.basis_r,op2.basis_r),(op1.operators...,op2))
LazyDirectSum(op1::AbstractOperator, op2::LazyDirectSum) = LazyDirectSum(directsum(op1.basis_l,op2.basis_l),directsum(op1.basis_r,op2.basis_r),(op1,op2.operators...))
LazyDirectSum(op1::LazyDirectSum, op2::LazyDirectSum) = LazyDirectSum(directsum(op1.basis_l,op2.basis_l),directsum(op1.basis_r,op2.basis_r),(op1.operators...,op2.operators...))
LazyDirectSum(op::AbstractOperator...) = reduce(LazyDirectSum, op)

function Base.:(==)(op1::LazyDirectSum,op2::LazyDirectSum)
    (op1.basis_l == op2.basis_l && op1.basis_r == op2.basis_r) || return false
    length(op1.operators)==length(op2.operators) || return false
    for (o1,o2) = zip(op1.operators,op2.operators)
        (o1 == o2) || return false
    end
    return true
end

# Algebra
dense(x::LazyDirectSum) = directsum(dense.(x.operators)...)
*(op1::LazyDirectSum{B1,B2},op2::LazyDirectSum{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}= LazyDirectSum(op1.basis_l,op2.basis_r,Tuple(op1.operators[i]*op2.operators[i] for i=1:length(op1.operators)))
-(op::LazyDirectSum) = LazyDirectSum([-op.operators[1];op.operators[2:end]]...)
+(op1::LazyDirectSum{B1,B2},op2::LazyDirectSum{B1,B2}) where {B1<:Basis,B2<:Basis} = LazyDirectSum((op1.operators .+ op2.operators)...)
dagger(op::LazyDirectSum) = LazyDirectSum(op.basis_r, op.basis_l, dagger.(op.operators))
Base.eltype(x::LazyDirectSum) = mapreduce(eltype, promote_type, x.operators)

directsum(op1::AbstractOperator,op2::LazyDirectSum) = LazyDirectSum(op1,op2)
directsum(op1::LazyDirectSum,op2::AbstractOperator) = LazyDirectSum(op1,op2)
directsum(op1::LazyDirectSum,op2::LazyDirectSum) = LazyDirectSum(op1,op2)

# Use lazy sum for FFTOperator
transform(b1::SumBasis,b2::SumBasis; kwargs...) = LazyDirectSum([transform(b1.bases[i],b2.bases[i];kwargs...) for i=1:length(b1.bases)]...)
directsum(op1::FFTOperator, op2::FFTOperator) = LazyDirectSum(op1,op2)

# Lazy embed
function embed(basis_l::SumBasis, basis_r::SumBasis, indices, op::LazyDirectSum)
    bases_l = basis_l.bases
    bases_r = basis_r.bases
    N = length(bases_r)
    @assert length(bases_l)==N

    T = eltype(op)

    function _get_op(i)
        idx = findfirst(isequal(i), indices)
        if idx === nothing
            return SparseOperator(T, bases_l[i], bases_r[i])
        else
            return op.operators[idx]
        end
    end

    ops = Tuple(_get_op(i) for i=1:N)
    return LazyDirectSum(basis_l,basis_r,ops)
end
# TODO: embed for multiple LazyDirectums?

# Fast in-place multiplication
function mul!(result::Ket{B1},M::LazyDirectSum{B1,B2},b::Ket{B2},alpha_,beta_) where {B1,B2}
    alpha = convert(ComplexF64, alpha_)
    beta = convert(ComplexF64, beta_)
    bases_r = b.basis.bases
    bases_l = M.basis_l.bases
    index = cumsum([0;length.(bases_r)...])
    for i=1:length(index)-1
        tmpket = Ket(bases_r[i], b.data[index[i]+1:index[i+1]])
        tmpresult = Ket(bases_l[i], result.data[index[i]+1:index[i+1]])
        mul!(tmpresult,M.operators[i],tmpket,alpha,beta)
        result.data[index[i]+1:index[i+1]] = tmpresult.data
    end
    return result
end
function mul!(result::Bra{B2},b::Bra{B1},M::LazyDirectSum{B1,B2},alpha_,beta_) where {B1,B2}
    alpha = convert(ComplexF64, alpha_)
    beta = convert(ComplexF64, beta_)
    bases_l = b.basis.bases
    bases_r = M.basis_r.bases
    index = cumsum([0;length.(bases_r)...])
    for i=1:length(index)-1
        tmpbra = Bra(bases_l[i], b.data[index[i]+1:index[i+1]])
        tmpresult = Bra(bases_r[i], result.data[index[i]+1:index[i+1]])
        mul!(tmpresult,tmpbra,M.operators[i],alpha,beta)
        result.data[index[i]+1:index[i+1]] = tmpresult.data
    end
    return result
end
