import Base: isequal, ==, *, /, +, -
import SparseArrays: sparse, spzeros

function _check_bases(basis_l, basis_r, operators)
    for o in operators
        basis_l == o.basis_l || throw(IncompatibleBases())
        basis_r == o.basis_r || throw(IncompatibleBases())
    end
end

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
    function LazySum(basis_l::BL,basis_r::BR,factors::F, operators::T; skip_checks=false) where {BL,BR,F,T}
        if !skip_checks
            length(operators)==length(factors) || throw(ArgumentError("LazySum `operators` and `factors` have different lengths."))
            _check_bases(basis_l, basis_r, operators)
        end
        new{BL,BR,F,T}(basis_l,basis_r,factors,operators)
    end
end

LazySum(::Type{Tf}, basis_l::Basis, basis_r::Basis) where Tf = LazySum(basis_l,basis_r,Tf[],())
LazySum(basis_l::Basis, basis_r::Basis) = LazySum(ComplexF64, basis_l, basis_r)

function LazySum(::Type{Tf}, factors, operators) where Tf
    if operators isa AbstractVector
        @info "LazySum operators with vector storage of operators may not perform well in time evolution."
    end
    factors_ = eltype(factors) != Tf ? Tf.(factors) : factors
    LazySum(operators[1].basis_l, operators[1].basis_r, factors_, operators)
end
function LazySum(factors, operators)
    Tf = promote_type(eltype(factors), mapreduce(eltype, promote_type, operators))
    LazySum(Tf, factors, operators)
end
LazySum(::Type{Tf}, operators::AbstractOperator...) where Tf = LazySum(ones(Tf, length(operators)), (operators...,))
LazySum(operators::AbstractOperator...) = LazySum(ComplexF64, operators...)
LazySum() = throw(ArgumentError("LazySum needs a basis, or at least one operator!"))

Base.copy(x::LazySum) = LazySum(x.basis_l, x.basis_r, copy(x.factors), copy.(x.operators); skip_checks=true)
Base.eltype(x::LazySum) = promote_type(eltype(x.factors), eltype.(x.operators)...)

dense(op::LazySum) = length(op.operators) > 0 ? sum(op.factors .* dense.(op.operators)) : Operator(op.basis_l, op.basis_r, zeros(eltype(op.factors), length(op.basis_l), length(op.basis_r)))
SparseArrays.sparse(op::LazySum) = length(op.operators) > 0 ? sum(op.factors .* sparse.(op.operators)) : Operator(op.basis_l, op.basis_r, spzeros(eltype(op.factors), length(op.basis_l), length(op.basis_r)))

isequal(x::LazySum, y::LazySum) = samebases(x,y) && isequal(x.operators, y.operators) && isequal(x.factors, y.factors)
==(x::LazySum, y::LazySum) = (samebases(x,y) && x.operators==y.operators && x.factors==y.factors)

_cat(opsA::Tuple, opsB::Tuple) = (opsA..., opsB...)
_cat(opsA::Tuple, opsB) = (opsA..., opsB...)
_cat(opsA, opsB::Tuple) = (opsA..., opsB...)
_cat(opsA, opsB) = vcat(opsA, opsB)

# Arithmetic operations
function +(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2}
    samebases(a,b) || throw(IncompatibleBases())
    factors = _cat(a.factors, b.factors)
    ops = _cat(a.operators, b.operators)
    LazySum(a.basis_l, a.basis_r, factors, ops; skip_checks=true)
end
+(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

-(a::LazySum) where {B1,B2,F,T} = LazySum(a.basis_l, a.basis_r, -a.factors, a.operators; skip_checks=true)
function -(a::LazySum{B1,B2}, b::LazySum{B1,B2}) where {B1,B2}
    samebases(a,b) || throw(IncompatibleBases())
    factors = _cat(a.factors, -b.factors)
    ops = _cat(a.operators, b.operators)
    LazySum(a.basis_l, a.basis_r, factors, ops; skip_checks=true)
end
-(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

function *(a::LazySum, b::Number)
    factors = b*a.factors
    LazySum(a.basis_l, a.basis_r, factors, a.operators; skip_checks=true)
end
*(a::Number, b::LazySum) = b*a

function /(a::LazySum, b::Number)
    factors = a.factors/b
    LazySum(a.basis_l, a.basis_r, factors, a.operators; skip_checks=true)
end

function dagger(op::LazySum)
    ops = dagger.(op.operators)
    LazySum(op.basis_r, op.basis_l, conj.(op.factors), ops; skip_checks=true)
end

tr(op::LazySum) = sum(op.factors .* tr.(op.operators))

_ptrace(ops::AbstractVector, indices) = [ptrace(op_i, indices) for op_i in ops]
_ptrace(ops, indices) = ((ptrace(op_i, indices) for op_i in ops)...,)
function ptrace(op::LazySum, indices)
    check_ptrace_arguments(op, indices)
    #rank = length(op.basis_l.shape) - length(indices) #????
    LazySum(op.factors, _ptrace(op.operators, indices))
end

normalize!(op::LazySum) = (op.factors /= tr(op); op)

_permute(ops::AbstractVector, perm) = [permutesystems(op_i, perm) for op_i in ops]
_permute(ops, perm) = ((permutesystems(op_i, perm) for op_i in ops)...,)
function permutesystems(op::LazySum, perm)
    ops = _permute(op.operators, perm)
    bl = ops[1].basis_l
    br = ops[1].basis_r
    LazySum(bl, br, op.factors, ops; skip_checks=true)
end

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
