
abstract type AbstractTimeDependentOperator <: AbstractOperator end

set_time!(o::AbstractOperator, ::Number) = o

(o::AbstractTimeDependentOperator)(t::Number) = set_time!(o, t)

for func in (:basis, :length, :size, :tr, :normalize, :normalize!,
    :identityoperator, :one, :eltype, :ptrace)
    @eval $func(op::AbstractTimeDependentOperator) = $func(static_operator(op))
end

expect(op::AbstractTimeDependentOperator, x) = expect(static_operator(op), x)
expect(index::Integer, op::AbstractTimeDependentOperator, x) = expect(index, static_operator(op), x)
variance(op::AbstractTimeDependentOperator, x) = variance(static_operator(op), x)
variance(index::Integer, op::AbstractTimeDependentOperator, x) = variance(index, static_operator(op), x)

promote_rule(::Type{T}, ::Type{S}) where {T<:AbstractTimeDependentOperator,S<:AbstractOperator} = T
convert(::Type{T}, O::AbstractOperator) where {T<:AbstractTimeDependentOperator} = T(O)

"""
    TimeDependentSum(lazysum, coeffs; init_time=0.0)
    TimeDependentSum(::Type{Tf}, basis_l, basis_r; init_time=0.0)
    TimeDependentSum([::Type{Tf},] [basis_l,] [basis_r,] coeffs, operators; init_time=0.0)
    TimeDependentSum([::Type{Tf},] coeff1=>op1, coeff2=>op2, ...; init_time=0.0)

Lazy sum of operators with time-dependent coefficients. Wraps a `LazySum` `lazysum`,
adding a `current_time` (operator "clock") and a means of specifying time
coefficients as numbers or functions of time.

The coefficient type `Tf` may be specified explicitly.
Time-dependent coefficients will be converted to this type on evaluation.
"""
mutable struct TimeDependentSum{BL<:Basis,BR<:Basis,O<:LazySum,C,T<:Number} <: AbstractTimeDependentOperator
    basis_l::BL
    basis_r::BR
    static_op::O
    coefficients::C
    current_time::T
    function TimeDependentSum(lazysum::O, coeffs::C; init_time::T=0.0) where {O<:LazySum,C,T<:Number}
        length(coeffs) == length(lazysum.operators) || throw(ArgumentError("Number of coefficients does not match number of operators."))
        bl = lazysum.basis_l
        br = lazysum.basis_r
        new{typeof(bl), typeof(br), O, C, T}(bl, br, lazysum, coeffs, init_time)
    end
end

Base.copy(op::TimeDependentSum) = TimeDependentSum(copy(op.static_op), copy.(op.coefficients))

function ==(A::TimeDependentSum, B::TimeDependentSum)
    A.current_time == B.current_time && A.coefficients == B.coefficients && A.static_op == B.static_op
end

static_operator(o::TimeDependentSum) = o.static_op

current_time(o::TimeDependentSum) = o.current_time

function set_time!(o::TimeDependentSum, t::Number)
    o.current_time = t
    update_static_coefficients!(o.static_op, o.coefficients, t)
    set_time!(o.static_op.operators, t)
end

is_const(op::TimeDependentSum) = all(is_const(c) for c in op.coefficients)
is_const(c::Number) = true
is_const(c) = false

coefficient_type(o::TimeDependentSum) = coefficient_type(static_operator(o))
coefficient_type(o::LazySum) = eltype(o.factors)

function TimeDependentSum(::Type{Tf}, basis_l::Basis, basis_r::Basis; init_time<:Number=0.0) where Tf
    TimeDependentSum(LazySum(Tf, basis_l, basis_r), Tf[]; init_time)
end

function TimeDependentSum(::Type{Tf}, basis_l::Basis, basis_r::Basis, coeffs, operators; init_time<:Number=0.0) where Tf
    coeff_vec = ones(Tf, length(coeffs))
    ls = LazySum(basis_l, basis_r, coeff_vec, operators)
    TimeDependentSum(ls, coeffs; init_time)
end

function TimeDependentSum(::Type{Tf}, coeffs, operators; init_time<:Number=0.0) where Tf
    TimeDependentSum(Tf, operators[1].basis_l, operators[1].basis_r, coeffs, operators; init_time)
end

function TimeDependentSum(coeffs, operators; init_time<:Number=0.0)
    Tf = mapreduce(typeof, promote_type, eval_coefficients(coeffs, init_time))
    TimeDependentSum(Tf, coeffs, operators; init_time)
end

function TimeDependentSum(::Type{Tf}, args::Vararg{Pair}; init_time<:Number=0.0) where Tf
    cs, ops = zip(args...)
    TimeDependentSum(Tf, [cs...], [ops...]; init_time)
end

function TimeDependentSum(args::Vararg{Pair}; init_time<:Number=0.0)
    cs, ops = zip(args...)
    TimeDependentSum([cs...], [ops...]; init_time)
end

TimeDependentSum(op::LazySum; init_time<:Number=0.0) = TimeDependentSum(op, op.factors; init_time)
TimeDependentSum(op::AbstractOperator; init_time<:Number=0.0) = TimeDependentSum(LazySum(op); init_time)
TimeDependentSum(coefficient, op::AbstractOperator; init_time<:Number=0.0) = TimeDependentSum([coefficient], [op]; init_time)
TimeDependentSum(op::TimeDependentSum) = op

function update_static_coefficients!(o::LazySum, coeffs, t)
    o.factors .= eval_coefficients(eltype(o.factors), coeffs, t)
    return
end

function update_static_coefficients!(o::LazySum, coeffs::Vector, t)
    T = eltype(o.factors)
    for (k, coeff) in enumerate(coeffs)
        o.factors[k] = T(eval_coefficient(coeff, t))
    end
    return
end

@inline eval_coefficient(c, t::Number) = c(t)
@inline eval_coefficient(c::Number, ::Number) = c
@inline eval_coefficients(coeffs, t::Number) = eval_coefficient.(coeffs, t)
@inline eval_coefficients(::Type{T}, coeffs::AbstractVector, t::Number) where T = T[T(eval_coefficient(c, t)) for c in coeffs]

# This is needed to avoid allocations in some cases, modeled on map(f, t::Tuple)
@inline eval_coefficients(coeffs::Tuple{Any,}, t::Number)          = (eval_coefficient(coeffs[1], t),)
@inline eval_coefficients(coeffs::Tuple{Any, Any}, t::Number)      = (eval_coefficient(coeffs[1], t), eval_coefficient(coeffs[2], t))
@inline eval_coefficients(coeffs::Tuple{Any, Any, Any}, t::Number) = (eval_coefficient(coeffs[1], t), eval_coefficient(coeffs[2], t), eval_coefficient(coeffs[3], t))
@inline eval_coefficients(coeffs::Tuple, t::Number)                = (eval_coefficient(coeffs[1], t), eval_coefficients(Base.tail(coeffs), t)...)

@inline eval_coefficients(::Type{T}, coeffs::Tuple{Any,}, t::Number) where T          = (T(eval_coefficient(coeffs[1], t)),)
@inline eval_coefficients(::Type{T}, coeffs::Tuple{Any, Any}, t::Number) where T      = (T(eval_coefficient(coeffs[1], t)), T(eval_coefficient(coeffs[2], t)))
@inline eval_coefficients(::Type{T}, coeffs::Tuple{Any, Any, Any}, t::Number) where T = (T(eval_coefficient(coeffs[1], t)), T(eval_coefficient(coeffs[2], t)), T(eval_coefficient(coeffs[3], t)))
@inline eval_coefficients(::Type{T}, coeffs::Tuple, t::Number) where T                = (T(eval_coefficient(coeffs[1], t)), eval_coefficients(T, Base.tail(coeffs), t)...)


function embed(basis_l::CompositeBasis, basis_r::CompositeBasis, i, o::TimeDependentSum)
    TimeDependentSum(embed(basis_l, basis_r, i, static_operator(o)), o.coefficients; init_time=o.current_time)
end

################

"""
TimeDependentOperator(::Type{Tf}, coeffs, operators)
TimeDependentOperator(coeffs, operators)

Creates a TimeDependentOperator from a sequence of `operators`` and matching
coefficients `coeffs`. There must be one coefficient, either a function or
a constant, for each operator.

The data type of coefficients is inferred from the constant coefficients
and the return values of the coefficient functions at time `t=1.0`. All
coefficients are converted to this datatype during evolution.
"""

"""
    TimeDependentOperator(::Type{Tf}, args::Vararg{Pair}) where Tf
    TimeDependentOperator(args::Vararg{Pair})

Creates a TimeDependentOperator from a sequence of pairs, for example
`TimeDependentOperator(1.0=>op1, t->cos(omega*t)=>op2)`, where each pair
matches an operator with a coefficient (function or constant).
"""

"""
    TimeDependentOperator(ops::Vector{T}) where T<:TimeDependentOperator

Creates a TimeDependentOperator from a vector of TimeDependentOperators.
"""
