import Base: isequal, ==, *, /, +, -
import SparseArrays: sparse, spzeros

function _check_bases(basis_l, basis_r, operators)
    for o in operators
        basis_l == o.basis_l || throw(IncompatibleBases())
        basis_r == o.basis_r || throw(IncompatibleBases())
    end
end

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
mutable struct LazySum{BL,BR,F,T} <: AbstractOperator{BL,BR}
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
LazySum(operators::AbstractOperator...) = LazySum(ComplexF64, operators...)
LazySum() = throw(ArgumentError("LazySum needs a basis, or at least one operator!"))

Base.copy(x::LazySum) = @samebases LazySum(x.basis_l, x.basis_r, copy(x.factors), copy.(x.operators))
Base.eltype(x::LazySum) = promote_type(eltype(x.factors), mapreduce(eltype, promote_type, x.operators))

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
+(a::LazySum, b::AbstractOperator) = a + LazySum(b)
+(a::AbstractOperator, b::LazySum) = LazySum(a) + b

-(a::LazySum) = @samebases LazySum(a.basis_l, a.basis_r, -a.factors, a.operators)
function -(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2}
    check_samebases(a,b)
    factors = _cat(a.factors, -b.factors)
    ops = _cat(a.operators, b.operators)
    @samebases LazySum(a.basis_l, a.basis_r, factors, ops)
end
-(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
-(a::LazySum, b::AbstractOperator) = a - LazySum(b)
-(a::AbstractOperator, b::LazySum) = LazySum(a) - b

function *(a::LazySum, b::Number)
    factors = b*a.factors
    @samebases LazySum(a.basis_l, a.basis_r, factors, a.operators)
end
*(a::Number, b::LazySum) = b*a

function /(a::LazySum, b::Number)
    factors = a.factors/b
    @samebases LazySum(a.basis_l, a.basis_r, factors, a.operators)
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
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Integer, op::LazySum) = embed(basis_l, basis_r, [i], op)

# Fast in-place multiplication
function mul!(result::Ket{B1},a::LazySum{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2}
    if length(a.operators) == 0
        result.data .*= beta
    else
        mul!(result,a.operators[1],b,alpha*a.factors[1],beta)
        for i=2:length(a.operators)
            mul!(result,a.operators[i],b,alpha*a.factors[i],1)
        end
    end
    return result
end

function mul!(result::Bra{B2},a::Bra{B1},b::LazySum{B1,B2},alpha,beta) where {B1,B2}
    if length(b.operators) == 0
        result.data .*= beta
    else
        mul!(result,a,b.operators[1],alpha*b.factors[1],beta)
        for i=2:length(b.operators)
            mul!(result,a,b.operators[i],alpha*b.factors[i],1)
        end
    end
    return result
end

function mul!(result::Operator{B1,B3},a::LazySum{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3}
    if length(a.operators) == 0
        result.data .*= beta
    else
        mul!(result,a.operators[1],b,alpha*a.factors[1],beta)
        for i=2:length(a.operators)
            mul!(result,a.operators[i],b,alpha*a.factors[i],1)
        end
    end
    return result
end
function mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::LazySum{B2,B3},alpha,beta) where {B1,B2,B3}
    if length(b.operators) == 0
        result.data .*= beta
    else
        mul!(result,a,b.operators[1],alpha*b.factors[1],beta)
        for i=2:length(b.operators)
            mul!(result,a,b.operators[i],alpha*b.factors[i],1)
        end
    end
    return result
end
