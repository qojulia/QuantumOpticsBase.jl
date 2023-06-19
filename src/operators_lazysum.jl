import Base: isequal, ==, *, /, +, -
import SparseArrays: sparse, spzeros
import QuantumInterface: BASES_CHECK

function _check_bases(basis_l, basis_r, operators)
    for o in operators
        basis_l == o.basis_l || throw(IncompatibleBases())
        basis_r == o.basis_r || throw(IncompatibleBases())
    end
end

"""
Abstract class for all Lazy type operators ([`LazySum`](@ref), [`LazyProduct`](@ref), and [`LazyTensor`](@ref))
"""
abstract type LazyOperator{BL,BR} <: AbstractOperator{BL,BR} end

"""
    LazySum([Tf,] [factors,] operators)
    LazySum([Tf,] basis_l, basis_r, [factors,] [operators])

Lazy evaluation of sums of operators.

All operators have to be given in respect to the same bases. The field
`factors` accounts for an additional multiplicative factor for each operator
stored in the `operators` field.

The factor type `Tf` can be specified to avoid having to infer it from the
factors and operators themselves. All `factors` will be converted to type `Tf`.

The `operators` will be kept as is. It can be, for example, a `Tuple` or a
`Vector` of operators. Using a `Tuple` is recommended for runtime performance
of operator-state operations, such as simulating time evolution. A `Vector` can
reduce compile-time overhead when doing arithmetic on `LazySum`s, such as
summing many `LazySum`s together.
"""
mutable struct LazySum{BL,BR,F,T} <: LazyOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factors::F
    operators::T
    function LazySum(basis_l::BL, basis_r::BR, factors::F, operators::T) where {BL,BR,F,T}
        length(operators) == length(factors) || throw(ArgumentError("LazySum `operators` and `factors` have different lengths."))
        BASES_CHECK[] && _check_bases(basis_l, basis_r, operators)
        new{BL,BR,F,T}(basis_l,basis_r,factors,operators)
    end
end

LazySum(::Type{Tf}, basis_l::Basis, basis_r::Basis) where Tf = LazySum(basis_l,basis_r,Tf[],())
LazySum(basis_l::Basis, basis_r::Basis) = LazySum(ComplexF64, basis_l, basis_r)

function LazySum(::Type{Tf}, basis_l::Basis, basis_r::Basis, factors, operators) where Tf
    factors_ = eltype(factors) != Tf ? Tf.(factors) : factors
    LazySum(basis_l, basis_r, factors_, operators)
end
function LazySum(::Type{Tf}, factors, operators) where Tf
    LazySum(Tf, operators[1].basis_l, operators[1].basis_r, factors, operators)
end
function LazySum(factors, operators)
    Tf = promote_type(eltype(factors), mapreduce(eltype, promote_type, operators))
    LazySum(Tf, factors, operators)
end
LazySum(::Type{Tf}, operators::AbstractOperator...) where Tf = LazySum(ones(Tf, length(operators)), (operators...,))
LazySum(operators::AbstractOperator...) = LazySum(mapreduce(eltype, promote_type, operators), operators...)
LazySum() = throw(ArgumentError("LazySum needs a basis, or at least one operator!"))
LazySum(x::LazySum) = x

Base.copy(x::LazySum) = @samebases LazySum(x.basis_l, x.basis_r, copy(x.factors), copy.(x.operators))
Base.eltype(x::LazySum) = mapreduce(eltype, promote_type, x.operators; init=eltype(x.factors))

dense(op::LazySum) = length(op.operators) > 0 ? sum(op.factors .* dense.(op.operators)) : Operator(op.basis_l, op.basis_r, zeros(eltype(op.factors), length(op.basis_l), length(op.basis_r)))
SparseArrays.sparse(op::LazySum) = length(op.operators) > 0 ? sum(op.factors .* sparse.(op.operators)) : Operator(op.basis_l, op.basis_r, spzeros(eltype(op.factors), length(op.basis_l), length(op.basis_r)))

isequal(x::LazySum, y::LazySum) = samebases(x,y) && isequal(x.operators, y.operators) && isequal(x.factors, y.factors)
==(x::LazySum, y::LazySum) = (samebases(x,y) && x.operators==y.operators && x.factors==y.factors)

# Make tuples contagious in LazySum arithmetic, but preserve "vector-only" cases
_cat(opsA::Tuple, opsB::Tuple) = (opsA..., opsB...)
_cat(opsA::Tuple, opsB) = (opsA..., opsB...)
_cat(opsA, opsB::Tuple) = (opsA..., opsB...)
_cat(opsA, opsB) = vcat(opsA, opsB)

# Arithmetic operations
function +(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2}
    check_samebases(a,b)
    factors = _cat(a.factors, b.factors)
    ops = _cat(a.operators, b.operators)
    @samebases LazySum(a.basis_l, a.basis_r, factors, ops)
end
+(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
+(a::LazyOperator, b::AbstractOperator) = LazySum(a) + LazySum(b)
+(a::AbstractOperator, b::LazyOperator) = LazySum(a) + LazySum(b)
+(a::O1, b::O2) where {O1<:LazyOperator,O2<:LazyOperator} = LazySum(a) + LazySum(b)

-(a::LazySum) = @samebases LazySum(a.basis_l, a.basis_r, -a.factors, a.operators)
function -(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2}
    check_samebases(a,b)
    factors = _cat(a.factors, -b.factors)
    ops = _cat(a.operators, b.operators)
    @samebases LazySum(a.basis_l, a.basis_r, factors, ops)
end
-(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
-(a::LazyOperator, b::AbstractOperator) = LazySum(a) - LazySum(b)
-(a::AbstractOperator, b::LazyOperator) = LazySum(a) - LazySum(b)
-(a::O1, b::O2) where {O1<:LazyOperator,O2<:LazyOperator} = LazySum(a) - LazySum(b)


#*(a::LazySum{B1,B2}, b::AbstractOperator{B2,B3}) where {B1,B2,B3} = LazySum(a.basis_l, a.basis_r, a.factors, a.operators .* b)
#*(a::AbstractOperator{B1,B2}, b::LazySum{B2,B3}) where {B1,B2,B3} = LazySum(a.basis_l, b.basis_r, b.factors, a.operators .* b)

#*(a::Operator{B1,B2}, b::LazySum{B2,B3}) where {B1,B2,B3} = LazySum(a.basis_l, b.basis_r, b.factors, a.operators .* b)
#*(a::LazySum{B1,B2}, b::Operator{B2,B3}) where {B1,B2,B3} = LazySum(a.basis_l, b.basis_r, a.factors, a.operators .* b)


function Base.:*(a::LazySum{B1,B2}, b::O2) where {B1,B2,O2<:LazyOperator}
    c = Array{AbstractOperator}(undef,length(a.operators))
    for i in eachindex(a.operators)
        c[i] = LazyProduct(a.operators[i]) * b
    end
    LazySum(a.basis_l, b.basis_r, a.factors, (c...,))
end

function Base.:*(a::O1, b::LazySum{B1,B2}) where {B1,B2,O1<:LazyOperator}
    c = Array{AbstractOperator}(undef,length(b.operators))
    for i in eachindex(b.operators)
        c[i] = a * LazyProduct(b.operators[i])
    end
    LazySum(a.basis_l, b.basis_r, b.factors, (c...,))
end


function Base.:*(a::O1, b::O2) where {O1<:LazySum,O2<:LazySum}
    c = Array{AbstractOperator}(undef,length(a.operators)*length(b.operators))
    factors = similar(a.factors,length(a.operators)*length(b.operators))
    k = 1
    for i in eachindex(a.operators)
        for j in eachindex(b.operators)
            factors[k] = a.factors[i] * b.factors[j]
            c[k] = a.operators[i] * b.operators[j]
            k += 1
        end
    end
    LazySum(a.basis_l, b.basis_r, factors, (c...,))
end

function *(a::LazySum, b::Number)
    factors = b*a.factors
    @samebases LazySum(a.basis_l, a.basis_r, factors, a.operators)
end
*(a::Number, b::LazySum) = b*a

function /(a::LazySum, b::Number)
    factors = a.factors/b
    @samebases LazySum(a.basis_l, a.basis_r, factors, a.operators)
end

function tensor(a::Operator,b::LazySum)
    btotal_l = a.basis_l ⊗ b.basis_l
    btotal_r = a.basis_r ⊗ b.basis_r
    ops = ([a ⊗ op for op in b.operators]...,)
    LazySum(btotal_l,btotal_r,b.factors,ops)
end
function tensor(a::LazySum,b::Operator)
    btotal_l = a.basis_l ⊗ b.basis_l
    btotal_r = a.basis_r ⊗ b.basis_r
    ops = ([op ⊗ b for op in a.operators]...,)
    LazySum(btotal_l,btotal_r,a.factors,ops)
end

function dagger(op::LazySum)
    ops = dagger.(op.operators)
    @samebases LazySum(op.basis_r, op.basis_l, conj.(op.factors), ops)
end

tr(op::LazySum) = sum(op.factors .* tr.(op.operators))

function ptrace(op::LazySum, indices)
    check_ptrace_arguments(op, indices)
    #rank = length(op.basis_l.shape) - length(indices) #????
    LazySum(op.factors, map(o->ptrace(o, indices), op.operators))
end

normalize!(op::LazySum) = (op.factors /= tr(op); op)

function permutesystems(op::LazySum, perm)
    ops = map(o->permutesystems(o, perm), op.operators)
    bl = ops[1].basis_l
    br = ops[1].basis_r
    @samebases LazySum(bl, br, op.factors, ops)
end

identityoperator(::Type{<:LazySum}, ::Type{S}, b1::Basis, b2::Basis) where S<:Number = LazySum(identityoperator(S, b1, b2))

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, op::LazySum)
    LazySum(basis_l, basis_r, op.factors, ((embed(basis_l, basis_r, indices, o) for o in op.operators)...,))
end
function embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, op::LazySum) # defined to avoid method ambiguity
    LazySum(basis_l, basis_r, op.factors, ((embed(basis_l, basis_r, index, o) for o in op.operators)...,)) # dispatch to fast-path single-index `embed`
end

function _zero_op_mul!(data, beta)
    if iszero(beta)
        fill!(data, zero(eltype(data)))
    elseif !isone(beta)
        data .*= beta
    end
    return data
end

# Fast in-place multiplication
function mul!(result::Ket{B1},a::LazySum{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2}
    if length(a.operators) == 0 || iszero(alpha)
        _check_mul!_dim_compatibility(size(result), size(a), size(b))
        _zero_op_mul!(result.data, beta)
    else
        mul!(result,a.operators[1],b,alpha*a.factors[1],beta)
        for i=2:length(a.operators)
            mul!(result,a.operators[i],b,alpha*a.factors[i],1)
        end
    end
    return result
end

function mul!(result::Bra{B2},a::Bra{B1},b::LazySum{B1,B2},alpha,beta) where {B1,B2}
    if length(b.operators) == 0 || iszero(alpha)
        _check_mul!_dim_compatibility(size(result), reverse(size(b)), size(a))
        _zero_op_mul!(result.data, beta)
    else
        mul!(result,a,b.operators[1],alpha*b.factors[1],beta)
        for i=2:length(b.operators)
            mul!(result,a,b.operators[i],alpha*b.factors[i],1)
        end
    end
    return result
end

function mul!(result::Operator{B1,B3},a::LazySum{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3}
    if length(a.operators) == 0 || iszero(alpha)
        _check_mul!_dim_compatibility(size(result), size(a), size(b))
        _zero_op_mul!(result.data, beta)
    else
        mul!(result,a.operators[1],b,alpha*a.factors[1],beta)
        for i=2:length(a.operators)
            mul!(result,a.operators[i],b,alpha*a.factors[i],1)
        end
    end
    return result
end
function mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::LazySum{B2,B3},alpha,beta) where {B1,B2,B3}
    if length(b.operators) == 0 || iszero(alpha)
        _check_mul!_dim_compatibility(size(result), size(a), size(b))
        _zero_op_mul!(result.data, beta)
    else
        mul!(result,a,b.operators[1],alpha*b.factors[1],beta)
        for i=2:length(b.operators)
            mul!(result,a,b.operators[i],alpha*b.factors[i],1)
        end
    end
    return result
end
