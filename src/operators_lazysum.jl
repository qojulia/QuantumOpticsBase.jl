import Base: isequal, ==, *, /, +, -
import SparseArrays: sparse, spzeros

"""
    LazySum([factors,] operators)
    LazySum(basis_l, basis_r, [factors,] [operators])

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
        length(operators)==length(factors) || throw(ArgumentError("LazySum `operators` and `factors` have different lengths."))
        for o in operators
            basis_l == o.basis_l || throw(IncompatibleBases())
            basis_r == o.basis_r || throw(IncompatibleBases())
        end
        new(basis_l, basis_r, factors, operators)
    end
end
LazySum(basis_l::BL,basis_r::BR,factors::F,operators::T) where {BL,BR,F,T} =
    LazySum{BL,BR,F,T}(basis_l,basis_r,factors,operators)

LazySum(::Type{T}, basis_l::Basis, basis_r::Basis) where T = LazySum(basis_l,basis_r,T[],())
LazySum(basis_l::Basis, basis_r::Basis) = LazySum(ComplexF64, basis_l, basis_r)

function LazySum(::Type{Tf}, factors, operators) where Tf
    factors_ = Tf.(factors)
    LazySum(operators[1].basis_l, operators[1].basis_r, factors_, operators)
end
function LazySum(factors, operators)
    Tf = promote_type(eltype(factors), mapreduce(eltype, promote_type, AbstractOperator[operators...]))
    LazySum(Tf, factors, operators)
end
LazySum(operators::AbstractOperator...) = LazySum(ones(ComplexF64, length(operators)), (operators...,))
LazySum(factors, operators::Vector) = LazySum(factors, (operators...,))
LazySum() = throw(ArgumentError("LazySum needs a basis, or at least one operator!"))

Base.copy(x::LazySum) = LazySum(copy(x.factors), ([copy(op) for op in x.operators]...,))
Base.eltype(x::LazySum) = promote_type(eltype(x.factors), eltype.(x.operators)...)

dense(op::LazySum) = length(op.operators) > 0 ? sum(op.factors .* dense.(op.operators)) : Operator(op.basis_l, op.basis_r, zeros(eltype(op.factors), length(op.basis_l), length(op.basis_r)))
SparseArrays.sparse(op::LazySum) = length(op.operators) > 0 ? sum(op.factors .* sparse.(op.operators)) : Operator(op.basis_l, op.basis_r, spzeros(eltype(op.factors), length(op.basis_l), length(op.basis_r)))

isequal(x::LazySum, y::LazySum) = samebases(x,y) && isequal(x.operators, y.operators) && isequal(x.factors, y.factors)
==(x::LazySum, y::LazySum) = (samebases(x,y) && x.operators==y.operators && x.factors==y.factors)

# Arithmetic operations
function +(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2}
    factors = [a.factors; b.factors]
    LazySum(eltype(factors), factors, (a.operators..., b.operators...))
end
+(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

-(a::LazySum) = LazySum(eltype(a.factors), -a.factors, a.operators)
function -(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2} 
    factors = [a.factors; -b.factors]
    LazySum(eltype(factors), factors, (a.operators..., b.operators...))
end
-(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

function *(a::LazySum, b::Number) 
    factors = b*a.factors
    LazySum(eltype(factors), factors, a.operators)
end
*(a::Number, b::LazySum) = b*a

function /(a::LazySum, b::Number) 
    factors = a.factors/b
    LazySum(eltype(factors), factors, a.operators)
end

dagger(op::LazySum) = LazySum(eltype(op.factors), conj.(op.factors), dagger.(op.operators))

tr(op::LazySum) = sum(op.factors .* tr.(op.operators))

function ptrace(op::LazySum, indices)
    check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    D = ([ptrace(op_i, indices) for op_i in op.operators]...,)
    LazySum(op.factors, D)
end

normalize!(op::LazySum) = (op.factors /= tr(op); op)

permutesystems(op::LazySum, perm) = LazySum(eltype(op.factors), op.factors, ([permutesystems(op_i, perm) for op_i in op.operators]...,))

identityoperator(::Type{<:LazySum}, ::Type{S}, b1::Basis, b2::Basis) where S<:Number = LazySum(identityoperator(S, b1, b2))

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, op::LazySum)
    LazySum(basis_l, basis_r, op.factors, ((embed(basis_l, basis_r, indices, o) for o in op.operators)...,))
end

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
