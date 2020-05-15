"""
    SumBasis(b1, b2...)

Similar to [`CompositeBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
mutable struct SumBasis{S,B<:Tuple{Vararg{Basis}}} <: Basis
    shape::S
    bases::B
end
SumBasis(bases::B) where B<:Tuple{Vararg{Basis}} = SumBasis(Int[length(b) for b in bases], bases)
SumBasis(shape::Vector{Int}, bases::Vector{B}) where B<:Basis = (tmp = (bases...,); SumBasis(shape, tmp))
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

const ⊕ = directsum

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
    getblock(x::Ket{<:SumBasis}, i::Int)

For a [`Ket`](@ref) defined on a [`SumBasis`](@ref), get the state as it is defined
on the ith sub-basis.
"""
function getblock(x::Ket{B}, i::Int) where B<:SumBasis
    b_i = x.basis.bases[i]
    inds = cumsum([0;length.(x.basis.bases[1:i])...])
    return Ket(b_i, x.data[inds[i]+1:inds[i+1]])
end

"""
    setblock!(x::Ket{<:SumBasis}, val::Ket, i::Int)

Set the data of `x` on the ith sub-basis equal to the data of `val`.
"""
function setblock!(x::Ket{B}, val::Ket, i::Int) where B<:SumBasis
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
directsum(a::DataOperator, b::DataOperator) = Operator(directsum(a.basis_l, b.basis_l), directsum(a.basis_r, b.basis_r), hcat([a.data; zeros(size(b))], [zeros(size(a)); b.data]))
directsum(a::AbstractOperator...) = reduce(directsum, a)

"""
    setblock!(op::DataOperator{<:SumBasis,<:SumBasis}, val::DataOperator, i::Int, j::Int)

Set the data of `op` corresponding to the block `(i,j)` equal to the data of `val`.
"""
function setblock!(op::DataOperator{<:SumBasis,<:SumBasis}, val::DataOperator, i::Int, j::Int)
    (bases_l,bases_r) = op.basis_l.bases, op.basis_r.bases
    check_samebases(bases_l[i], val.basis_l)
    check_samebases(bases_r[j], val.basis_r)
    inds_i = cumsum([0;length.(bases_l[1:i])...])
    inds_j = cumsum([0;length.(bases_r[1:j])...])
    op.data[inds_i[i]+1:inds_i[i+1],inds_j[j]+1:inds_j[j+1]] = val.data
    return op
end

"""
    getblock(op::Operator{<:SumBasis,<:SumBasis}, i::int, j::Int)

Get the sub-basis operator corresponding to the block `(i,j)` of `op`.
"""
function getblock(op::DataOperator{BL,BR}, i::Int, j::Int) where {BL<:SumBasis,BR<:SumBasis}
    (bases_l,bases_r) = op.basis_l.bases, op.basis_r.bases
    inds_i = cumsum([0;length.(bases_l[1:i])...])
    inds_j = cumsum([0;length.(bases_r[1:j])...])
    data = op.data[inds_i[i]+1:inds_i[i+1],inds_j[j]+1:inds_j[j+1]]
    return Operator(bases_l[i],bases_r[j], data)
end

"""
    LazyDirectSum <: AbstractOperator

Lazy implementation of `directsum`
"""
mutable struct LazyDirectSum{BL<:SumBasis, BR<:SumBasis, T<:Tuple{Vararg{AbstractOperator}}} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    operators::T
end
LazyDirectSum(op1::AbstractOperator, op2::AbstractOperator) = LazyDirectSum(directsum(op1.basis_l,op2.basis_l),directsum(op1.basis_r,op2.basis_r),(op1,op2))
LazyDirectSum(op::AbstractOperator...) = reduce(LazyDirectSum, op)
dense(x::LazyDirectSum) = directsum(dense.(x.operators)...)
*(op1::LazyDirectSum{B1,B2},op2::LazyDirectSum{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}= LazyDirectSum(op1.basis_l,op2.basis_r,Tuple(op1.operators[i]*op2.operators[i] for i=1:length(op1.operators)))
-(op::LazyDirectSum) = LazyDirectSum([-op.operators[1];op.operators[2:end]]...)
+(op1::LazyDirectSum{B1,B2},op2::LazyDirectSum{B1,B2}) where {B1<:Basis,B2<:Basis} = LazyDirectSum((op1.operators .+ op2.operators)...)

# Use lazy sum for FFTOperator
transform(b1::SumBasis,b2::SumBasis; kwargs...) = LazyDirectSum([transform(b1.bases[i],b2.bases[i];kwargs...) for i=1:length(b1.bases)]...)
directsum(op1::FFTOperator, op2::FFTOperator) = LazyDirectSum(op1,op2)


# Fast in-place multiplication
function mul!(result::Ket{B1},M::LazyDirectSum{B1,B2},b::Ket{B2},alpha_,beta_) where {B1<:SumBasis,B2<:SumBasis}
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
function mul!(result::Bra{B2},b::Bra{B1},M::LazyDirectSum{B1,B2},alpha_,beta_) where {B1<:SumBasis,B2<:SumBasis}
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
