import Base: ==, *, /, +, -
import SparseArrays: sparse

"""
    LazySum([factors,] operators)

Lazy evaluation of sums of operators.

All operators have to be given in respect to the same bases. The field
`factors` accounts for an additional multiplicative factor for each operator
stored in the `operators` field.
"""
mutable struct LazySum{BL,BR,F,T} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factors::F
    operators::T
    function LazySum{BL,BR,F,T}(basis_l::BL,basis_r::BR,factors::F,operators::T) where {BL,BR,F,T}
        @assert length(operators)==length(factors)
        new(basis_l, basis_r, factors, operators)
    end
end
LazySum(basis_l::BL,basis_r::BR,factors::F,operators::T) where {BL,BR,F,T} =
    LazySum{BL,BR,F,T}(basis_l,basis_r,factors,operators)

function LazySum(factors, operators)
    for i = 2:length(operators)
        operators[1].basis_l == operators[i].basis_l || throw(IncompatibleBases())
        operators[1].basis_r == operators[i].basis_r || throw(IncompatibleBases())
    end
    Tf = promote_type(eltype(factors), eltype.(operators)...)
    factors_ = Tf.(factors)
    LazySum(operators[1].basis_l, operators[1].basis_r, factors_, operators)
end
LazySum(operators::AbstractOperator...) = LazySum(ones(ComplexF64, length(operators)), (operators...,))
LazySum(factors, operators::Vector) = LazySum(factors, (operators...,))
LazySum() = throw(ArgumentError("LazySum needs at least one operator!"))

Base.copy(x::LazySum) = LazySum(copy(x.factors), ([copy(op) for op in x.operators]...,))
Base.eltype(x::LazySum) = promote_type(eltype(x.factors), eltype.(x.operators)...)

dense(op::LazySum) = sum(op.factors .* dense.(op.operators))
SparseArrays.sparse(op::LazySum) = sum(op.factors .* sparse.(op.operators))

==(x::LazySum, y::LazySum) = (x.basis_l == y.basis_l && x.basis_r == y.basis_r && x.operators==y.operators && x.factors==y.factors)

# Arithmetic operations
+(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2} = LazySum([a.factors; b.factors], (a.operators..., b.operators...))
+(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

-(a::LazySum) = LazySum(-a.factors, a.operators)
-(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2} = LazySum([a.factors; -b.factors], (a.operators..., b.operators...))
-(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

*(a::LazySum, b::Number) = LazySum(b*a.factors, a.operators)
*(a::Number, b::LazySum) = LazySum(a*b.factors, b.operators)

/(a::LazySum, b::Number) = LazySum(a.factors/b, a.operators)

dagger(op::LazySum) = LazySum(conj.(op.factors), dagger.(op.operators))

tr(op::LazySum) = sum(op.factors .* tr.(op.operators))

function ptrace(op::LazySum, indices)
    check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    D = ([ptrace(op_i, indices) for op_i in op.operators]...,)
    LazySum(op.factors, D)
end

normalize!(op::LazySum) = (op.factors /= tr(op); op)

permutesystems(op::LazySum, perm) = LazySum(op.factors, ([permutesystems(op_i, perm) for op_i in op.operators]...,))

identityoperator(::Type{<:LazySum}, ::Type{S}, b1::Basis, b2::Basis) where S<:Number = LazySum(identityoperator(S, b1, b2))


# Fast in-place multiplication
function mul!(result::Ket{B1},a::LazySum{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2}
    mul!(result,a.operators[1],b,alpha*a.factors[1],beta)
    for i=2:length(a.operators)
        mul!(result,a.operators[i],b,alpha*a.factors[i],1)
    end
    return result
end

function mul!(result::Bra{B2},a::Bra{B1},b::LazySum{B1,B2},alpha,beta) where {B1,B2}
    mul!(result,a,b.operators[1],alpha*b.factors[1],beta)
    for i=2:length(b.operators)
        mul!(result,a,b.operators[i],alpha*b.factors[i],1)
    end
    return result
end

function mul!(result::Operator{B1,B3},a::LazySum{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3}
    mul!(result,a.operators[1],b,alpha*a.factors[1],beta)
    for i=2:length(a.operators)
        mul!(result,a.operators[i],b,alpha*a.factors[i],1)
    end
    return result
end
function mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::LazySum{B2,B3},alpha,beta) where {B1,B2,B3}
    mul!(result,a,b.operators[1],alpha*b.factors[1],beta)
    for i=2:length(b.operators)
        mul!(result,a,b.operators[i],alpha*b.factors[i],1)
    end
    return result
end
