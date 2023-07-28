
import Base: size, *, +, -, /, ==, isequal, adjoint, convert

"""
    AbstractTimeDependentOperator{BL,BR} <: AbstractOperator{BL,BR}

Abstract type providing a time-dependent operator interface. Time-dependent
operators have internal "clocks" that can be addressed with [`set_time!`](@ref)
and [`current_time`](@ref). A shorthand `op(t)`, equivalent to
`set_time!(copy(op), t)`, is available for brevity.

A time-dependent operator is always concrete-valued according to the current
time of its internal clock.
"""
abstract type AbstractTimeDependentOperator{BL,BR} <: AbstractOperator{BL,BR} end

"""
    current_time(op::AbstractOperator)

Returns the current time of the operator `op`. If `op` is not time-dependent,
this throws an `ArgumentError`.
"""
current_time(::T) where {T<:AbstractOperator} = 
  throw(ArgumentError("Time not defined for operators of type $T. Consider using a TimeDependentSum or another time-dependence wrapper."))
static_operator(o::AbstractOperator) = o

"""
    set_time!(o::AbstractOperator, t::Number)

Sets the clock of an operator (see [`AbstractTimeDependentOperator`](@ref)).
If `o` contains other operators (e.g. in case `o` is a `LazyOperator`),
recursively calls `set_time!` on those.

This does nothing in case `o` is not time-dependent.
"""
set_time!(o::AbstractOperator, ::Number) = o
set_time!(o::LazyOperator, t::Number) = (set_time!.(o.operators, t); return o)

(o::AbstractTimeDependentOperator)(t::Number) = set_time!(copy(o), t)

function _check_same_time(A::AbstractTimeDependentOperator, B::AbstractTimeDependentOperator)
    tA = current_time(A)
    tB = current_time(B)
    tA == tB || throw(ArgumentError(
        "Time-dependent operators with different times ($tA and $tB) cannot be combined. Consider setting their clocks to a common time with `set_time!`."))
end

is_const(op::AbstractTimeDependentOperator) = false
is_const(op::AbstractOperator) = true
is_const(op::LazyOperator) = all(is_const(o) for o in suboperators(op))
is_const(op::LazyTensor) = all(is_const(o) for o in op.operators)
is_const(c::Number) = true
is_const(c::Function) = false

for func in (:basis, :length, :size, :tr, :normalize, :normalize!,
    :identityoperator, :one, :eltype, :ptrace)
    @eval $func(op::AbstractTimeDependentOperator) = $func(static_operator(op))
end

for func in (:expect, :variance)
    @eval $func(op::AbstractTimeDependentOperator{B,B}, x::Ket{B}) where B = $func(static_operator(op), x)
    @eval $func(op::AbstractTimeDependentOperator{B,B}, x::AbstractOperator{B,B}) where B = $func(static_operator(op), x)
    @eval $func(index::Integer, op::AbstractTimeDependentOperator{B1,B2}, x::AbstractOperator{B3,B3}) where {B1,B2,B3<:CompositeBasis} = $func(index, static_operator(op), x)
    @eval $func(indices, op::AbstractTimeDependentOperator{B1,B2}, x::AbstractOperator{B3,B3}) where {B1,B2,B3<:CompositeBasis} = $func(indices, static_operator(op), x)
end

# TODO: Consider using promotion to define arithmetic between operator types
#promote_rule(::Type{T}, ::Type{S}) where {T<:AbstractTimeDependentOperator,S<:AbstractOperator} = T
#convert(::Type{T}, O::AbstractOperator) where {T<:AbstractTimeDependentOperator} = T(O)

const VecOrTuple = Union{Tuple,AbstractVector}

"""
    TimeDependentSum(lazysum, coeffs, init_time)
    TimeDependentSum(::Type{Tf}, basis_l, basis_r; init_time=0.0)
    TimeDependentSum([::Type{Tf},] [basis_l,] [basis_r,] coeffs, operators; init_time=0.0)
    TimeDependentSum([::Type{Tf},] coeff1=>op1, coeff2=>op2, ...; init_time=0.0)
    TimeDependentSum(::Tuple, op::TimeDependentSum)

Lazy sum of operators with time-dependent coefficients. Wraps a
[`LazySum`](@ref) `lazysum`, adding a `current_time` (or operator "clock") and a
means of specifying time coefficients as functions of time (or numbers).

The coefficient type `Tf` may be specified explicitly.
Time-dependent coefficients will be converted to this type on evaluation.
"""
mutable struct TimeDependentSum{BL<:Basis,BR<:Basis,C<:VecOrTuple,O<:LazySum,T<:Number} <: AbstractTimeDependentOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    coefficients::C
    static_op::O
    current_time::T
    function TimeDependentSum(coeffs::C, lazysum::O, init_time::T) where {C<:VecOrTuple,O<:LazySum,T<:Number}
        length(coeffs) == length(lazysum.operators) || throw(ArgumentError("Number of coefficients does not match number of operators."))
        bl = lazysum.basis_l
        br = lazysum.basis_r
        update_static_coefficients!(lazysum, coeffs, init_time)
        set_time!(lazysum, init_time)
        new{typeof(bl), typeof(br), C, O, T}(bl, br, coeffs, lazysum, init_time)
    end
end
TimeDependentSum(coeffs::C, lazysum::O; init_time::T=0.0) where {C<:VecOrTuple,O<:LazySum,T<:Number} = TimeDependentSum(coeffs, lazysum, init_time)

function TimeDependentSum(::Type{Tf}, basis_l::Basis, basis_r::Basis; init_time::T=0.0) where {Tf<:Number,T<:Number}
    TimeDependentSum(Tf[], LazySum(Tf, basis_l, basis_r), init_time)
end

function TimeDependentSum(::Type{Tf}, basis_l::Basis, basis_r::Basis, coeffs::C, operators::O; init_time::T=0.0) where {Tf<:Number,C<:VecOrTuple,O<:VecOrTuple,T<:Number}
    coeff_vec = ones(Tf, length(coeffs))
    ls = LazySum(basis_l, basis_r, coeff_vec, operators)
    TimeDependentSum(coeffs, ls, init_time)
end

function TimeDependentSum(::Type{Tf}, coeffs::C, operators::O; init_time::T=0.0) where {Tf<:Number,C<:VecOrTuple,O<:VecOrTuple,T<:Number}
    TimeDependentSum(Tf, operators[1].basis_l, operators[1].basis_r, coeffs, operators; init_time)
end

function TimeDependentSum(coeffs::C, operators::O; init_time::T=0.0)  where {C<:VecOrTuple,O<:VecOrTuple,T<:Number}
    Tf = mapreduce(typeof, promote_type, eval_coefficients(coeffs, init_time))
    TimeDependentSum(Tf, coeffs, operators; init_time)
end

function TimeDependentSum(::Type{Tf}, args::Vararg{Pair}; init_time::T=0.0) where {Tf<:Number,T<:Number}
    cs, ops = zip(args...)
    TimeDependentSum(Tf, [cs...], [ops...]; init_time)
end

function TimeDependentSum(args::Vararg{Pair}; init_time::T=0.0) where {T<:Number}
    cs, _ = zip(args...)
    Tf = mapreduce(typeof, promote_type, eval_coefficients(cs, init_time))
    TimeDependentSum(Tf, args...; init_time)
end

TimeDependentSum(op::LazySum; init_time::Number=0.0) = TimeDependentSum(op.factors, op, init_time)
TimeDependentSum(op::AbstractOperator; init_time::Number=0.0) = TimeDependentSum(LazySum(op); init_time)
TimeDependentSum(coefficient, op::AbstractOperator; init_time::Number=0.0) = TimeDependentSum([coefficient], [op]; init_time)
TimeDependentSum(op::TimeDependentSum) = op
TimeDependentSum(::Type{Tuple}, op::TimeDependentSum) = TimeDependentSum((coefficients(op)...,), LazySum(Tuple, static_operator(op)), current_time(op))

static_operator(o::TimeDependentSum) = o.static_op
coefficients(o::TimeDependentSum) = o.coefficients
current_time(o::TimeDependentSum) = o.current_time
suboperators(o::TimeDependentSum) = static_operator(o).operators
eval_coefficients(o::TimeDependentSum, t::Number) = eval_coefficients(coefficient_type(o), coefficients(o), t)

function set_time!(o::TimeDependentSum, t::Number)
    if o.current_time != t
        o.current_time = t
        update_static_coefficients!(static_operator(o), coefficients(o), t)
    end
    set_time!.(suboperators(o), t)
    o
end

is_const(op::TimeDependentSum) = all(is_const(c) for c in op.coefficients) && all(is_const(o) for o in suboperators(op))

coefficient_type(o::TimeDependentSum) = coefficient_type(static_operator(o))
coefficient_type(o::LazySum) = eltype(o.factors)

_coeff_copy(c) = copy(c)
_coeff_copy(c::Function) = c
_coeff_copy(t::Tuple) = _coeff_copy.(t)
Base.copy(op::TimeDependentSum) = TimeDependentSum(_coeff_copy(op.coefficients), copy(op.static_op); init_time=current_time(op))

function ==(A::TimeDependentSum, B::TimeDependentSum)
    A.current_time == B.current_time && A.coefficients == B.coefficients && A.static_op == B.static_op
end

function isequal(A::TimeDependentSum, B::TimeDependentSum)
    isequal(A.current_time, B.current_time) && isequal(A.coefficients, B.coefficients) && isequal(A.static_op, B.static_op)
end

_lazysum_op_map(f, op::LazySum) = LazySum(eltype(op.factors), op.basis_l, op.basis_r, copy.(op.factors), map(f, op.operators))

dense(op::TimeDependentSum) = TimeDependentSum(coefficients(op), _lazysum_op_map(dense, static_operator(op)), current_time(op))
sparse(op::TimeDependentSum) = TimeDependentSum(coefficients(op), _lazysum_op_map(sparse, static_operator(op)), current_time(op))

_conj_coeff(c) = conj(c)
_conj_coeff(c::Function) = conj ∘ c
function dagger(op::TimeDependentSum)
    TimeDependentSum(
        _conj_coeff.(coefficients(op)),
        dagger(static_operator(op)),
        current_time(op))
end
adjoint(op::TimeDependentSum) = dagger(op)

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis, i::Integer, o::TimeDependentSum)
    TimeDependentSum(coefficients(o), embed(basis_l, basis_r, i, static_operator(o)), o.current_time)
end

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis, indices, o::TimeDependentSum)
    TimeDependentSum(coefficients(o), embed(basis_l, basis_r, indices, static_operator(o)), o.current_time)
end

function +(A::TimeDependentSum, B::TimeDependentSum)
    _check_same_time(A, B)
    TimeDependentSum(
        _lazysum_cat(coefficients(A), coefficients(B)),
        static_operator(A) + static_operator(B),
        current_time(A))
end
+(A::AbstractOperator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) + B
+(A::TimeDependentSum, B::AbstractOperator) = A + TimeDependentSum(B; init_time=current_time(A))
+(A::LazyOperator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) + B
+(A::TimeDependentSum, B::LazyOperator) = A + TimeDependentSum(B; init_time=current_time(A))
+(A::Operator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) + B
+(A::TimeDependentSum, B::Operator) = A + TimeDependentSum(B; init_time=current_time(A))

_unary_minus(c::Function) = Base.:- ∘ c
_unary_minus(c) = -c
function -(o::TimeDependentSum)
    TimeDependentSum(_unary_minus.(coefficients(o)), -static_operator(o), current_time(o))
end

function -(A::TimeDependentSum, B::TimeDependentSum)
    _check_same_time(A, B)
    TimeDependentSum(
        _lazysum_cat(coefficients(A), _unary_minus.(coefficients(B))),
        static_operator(A) - static_operator(B),
        current_time(A))
end
-(A::AbstractOperator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) - B
-(A::TimeDependentSum, B::AbstractOperator) = A - TimeDependentSum(B; init_time=current_time(A))
-(A::LazyOperator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) - B
-(A::TimeDependentSum, B::LazyOperator) = A - TimeDependentSum(B; init_time=current_time(A))
-(A::Operator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) - B
-(A::TimeDependentSum, B::Operator) = A - TimeDependentSum(B; init_time=current_time(A))

_mul_coeffs(a, b) = a*b
_mul_coeffs(a, b::Function) = (@inline multiplied_coeffs_fn(t)=a*b(t))
_mul_coeffs(a::Function, b) = _mul_coeffs(b, a)
_mul_coeffs(a::Function, b::Function) = (@inline multiplied_coeffs_ff(t)=a(t)*b(t))
function *(A::TimeDependentSum, B::TimeDependentSum)
    _check_same_time(A, B)
    coeffs = _lazysum_cartprod(_mul_coeffs, coefficients(A), coefficients(B))
    TimeDependentSum(coeffs, static_operator(A) * static_operator(B), current_time(A))
end
*(A::AbstractOperator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) * B
*(A::TimeDependentSum, B::AbstractOperator) = A * TimeDependentSum(B; init_time=current_time(A))
*(A::LazyOperator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) * B
*(A::TimeDependentSum, B::LazyOperator) = A * TimeDependentSum(B; init_time=current_time(A))
*(A::Operator, B::TimeDependentSum) = TimeDependentSum(A; init_time=current_time(B)) * B
*(A::TimeDependentSum, B::Operator) = A * TimeDependentSum(B; init_time=current_time(A))

function *(A::TimeDependentSum, B::Number)
    TimeDependentSum(_mul_coeffs.(coefficients(A), B), static_operator(A) * B, current_time(A))
end
*(A::Number, B::TimeDependentSum) = B*A

_div_coeffs(a, b) = a/b
_div_coeffs(a::Function, b) = _mul_coeffs(a, 1/b)
function /(A::TimeDependentSum, B::Number)
    TimeDependentSum(_div_coeffs.(coefficients(A), B), static_operator(A) / B, current_time(A))
end

mul!(out::Operator{B1,B3}, a::TimeDependentSum{B1,B2}, b::Operator{B2,B3}, alpha, beta) where {B1,B2,B3} = mul!(out, static_operator(a), b, alpha, beta)
mul!(out::Ket{B1}, a::TimeDependentSum{B1,B2}, b::Ket{B2}, alpha, beta) where {B1,B2} = mul!(out, static_operator(a), b, alpha, beta)
mul!(out::Operator{B1,B3}, a::Operator{B1,B2}, b::TimeDependentSum{B2,B3}, alpha, beta) where {B1,B2,B3} = mul!(out, a, static_operator(b), alpha, beta)
mul!(out::Bra{B2}, a::Bra{B1}, b::TimeDependentSum{B1,B2}, alpha, beta) where {B1,B2} = mul!(out, a, static_operator(b), alpha, beta)

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
@inline eval_coefficients(coeffs::AbstractVector, t::Number) = [eval_coefficient(c, t) for c in coeffs]
@inline eval_coefficients(::Type{T}, coeffs::AbstractVector, t::Number) where T = T[T(eval_coefficient(c, t)) for c in coeffs]

@inline eval_coefficients(coeffs::Tuple, t::Number) = map(c->eval_coefficient(c, t), coeffs)

# This is the performance-critical implementation.
# To avoid allocations in most cases, we model this on map(f, t::Tuple).
@inline eval_coefficients(::Type{T}, coeffs::Tuple{Any,}, t::Number) where T          = (T(eval_coefficient(coeffs[1], t)),)
@inline eval_coefficients(::Type{T}, coeffs::Tuple{Any, Any}, t::Number) where T      = (T(eval_coefficient(coeffs[1], t)), T(eval_coefficient(coeffs[2], t)))
@inline eval_coefficients(::Type{T}, coeffs::Tuple{Any, Any, Any}, t::Number) where T = (T(eval_coefficient(coeffs[1], t)), T(eval_coefficient(coeffs[2], t)), T(eval_coefficient(coeffs[3], t)))
@inline eval_coefficients(::Type{T}, coeffs::Tuple, t::Number) where T                = (T(eval_coefficient(coeffs[1], t)), eval_coefficients(T, Base.tail(coeffs), t)...)


_timeshift_coeff(coeff, t0) = (@inline shifted_coeff(t) = coeff(t-t0))
_timeshift_coeff(coeff::Number, _) = coeff

"""
    time_shift(op::TimeDependentSum, t0)

Shift (translate) a [`TimeDependentSum`](@ref) `op` forward in time (delaying its
action) by `t0` units, so that the coefficient functions of time `f(t)` become
`f(t-t0)`. Return a new [`TimeDependentSum`](@ref).
"""
function time_shift(op::TimeDependentSum, t0)
    iszero(t0) && return op
    TimeDependentSum(_timeshift_coeff.(coefficients(op), t0), copy(static_operator(op)), current_time(op))
end

_timestretch_coeff(coeff, Sfactor) = (@inline stretched_coeff(t) = coeff(t/Sfactor)) 
_timestretch_coeff(coeff::Number, _) = coeff

"""
    time_stretch(op::TimeDependentSum, Sfactor)
Stretch (in time) a [`TimeDependentSum`](@ref) `op` by a factor of `Sfactor` (making it 'longer'),
so that the coefficient functions of time `f(t)` become `f(t/Sfactor)`. Return a new [`TimeDependentSum`](@ref).
"""
function time_stretch(op::TimeDependentSum, Sfactor)
    isone(Sfactor) && return op
    TimeDependentSum(_timestretch_coeff.(coefficients(op), Sfactor), copy(static_operator(op)), current_time(op))
end

_restrict_coeff(c::Number, t_from, t_to) = (@inline restricted_coeff_n(t) = ifelse(t_from <= t < t_to, c, zero(c)))
_restrict_coeff(c, t_from, t_to) = (@inline restricted_coeff_f(t) = ifelse(t_from <= t < t_to, c(t), zero(c(t_from))))

"""
    time_restrict(op::TimeDependentSum, t_from, t_to)
    time_restrict(op::TimeDependentSum, t_to)

Restrict a [`TimeDependentSum`](@ref) `op` to the time window `t_from <= t < t_to`,
forcing it to be exactly zero outside that range of times. If `t_from` is not
provided, it is assumed to be zero.
Return a new [`TimeDependentSum`](@ref).
"""
function time_restrict(op::TimeDependentSum, t_from, t_to)
    TimeDependentSum(_restrict_coeff.(coefficients(op), t_from, t_to), copy(static_operator(op)), current_time(op))
end
time_restrict(op::TimeDependentSum, t_duration) = time_restrict(op, zero(t_duration), t_duration)
