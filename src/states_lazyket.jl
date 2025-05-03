"""
    LazyKet(b, kets)

Lazy implementation of a tensor product of kets.

The subkets are stored in the `kets` field.
The main purpose of such a ket are simple computations for large product states, such as expectation values.
It's used to compute numeric initial states in QuantumCumulants.jl (see QuantumCumulants.initial_values).
"""
mutable struct LazyKet{B,T} <: AbstractKet
    basis::B
    kets::T
    function LazyKet(b::B, kets::T) where {B<:CompositeBasis,T<:Tuple}
        N = length(b)
        for n=1:N
            @assert isa(kets[n], Ket)
            @assert basis(kets[n]) == b[n]
        end
        new{B,T}(b, kets)
    end
end
function LazyKet(b::CompositeBasis, kets::Vector)
    return LazyKet(b,Tuple(kets))
end

basis(ket::LazyKet) = ket.basis

Base.eltype(ket::LazyKet) = Base.promote_type(eltype.(ket.kets)...)

Base.isequal(x::LazyKet, y::LazyKet) = isequal(x.basis, y.basis) && isequal(x.kets, y.kets)
Base.:(==)(x::LazyKet, y::LazyKet) = (x.basis == y.basis) && (x.kets == y.kets)

# conversion to dense
Ket(ket::LazyKet) = ⊗(ket.kets...)

# no lazy bras for now
dagger(x::LazyKet) = throw(MethodError("dagger not implemented for LazyKet: LazyBra is currently not implemented at all!"))

# tensor with other kets
function tensor(x::LazyKet, y::Ket)
    return LazyKet(x.basis ⊗ y.basis, (x.kets..., y))
end
function tensor(x::Ket, y::LazyKet)
    return LazyKet(x.basis ⊗ y.basis, (x, y.kets...))
end
function tensor(x::LazyKet, y::LazyKet)
    return LazyKet(x.basis ⊗ y.basis, (x.kets..., y.kets...))
end

# norms
norm(state::LazyKet) = prod(norm.(state.kets))
function normalize!(state::LazyKet)
    for ket in state.kets
        normalize!(ket)
    end
    return state
end
function normalize(state::LazyKet)
    y = deepcopy(state)
    normalize!(y)
    return y
end

# expect
function expect(op::LazyTensor, state::LazyKet)
    check_multiplicable(op,op); check_multiplicable(op, state)
    ops = op.operators
    inds = op.indices
    kets = state.kets

    T = promote_type(eltype(op), eltype(state))
    exp_val = convert(T, op.factor)

    # loop over all operators and match with corresponding kets
    for (i, op_) in enumerate(op.operators)
        exp_val *= expect(op_, kets[inds[i]])
    end

    # loop over remaining kets and just add the norm (should be one for normalized ones, but hey, who knows..)
    for i in 1:length(kets)
        if i ∉ inds
            exp_val *= dot(kets[i].data, kets[i].data)
        end
    end

    return exp_val
end

function expect(op::LazyProduct, state::LazyKet)
    check_multiplicable(op,op); check_multiplicable(op, state)

    tmp_state1 = deepcopy(state)
    tmp_state2 = deepcopy(state)
    for i = length(op.operators):-1:1
        mul!(tmp_state2, op.operators[i], tmp_state1)
        for j = 1:length(state.kets)
            copyto!(tmp_state1.kets[j].data, tmp_state2.kets[j].data)
        end
    end

    T = promote_type(eltype(op), eltype(state))
    exp_val = convert(T, op.factor)
    for i = 1:length(state.kets)
        exp_val *= dot(state.kets[i].data, tmp_state2.kets[i].data)
    end

    return exp_val
end

function expect(op::LazySum, state::LazyKet)
    check_multiplicable(op,op); check_multiplicable(op, state)

    T = promote_type(eltype(op), eltype(state))
    exp_val = zero(T)
    for (factor, sub_op) in zip(op.factors, op.operators)
        exp_val += factor * expect(sub_op, state)
    end
   
    return exp_val
end


# mul! with lazytensor -- needed to compute lazyproduct averages (since ⟨op1 * op2⟩ doesn't factorize)
# this mul! is the only one that really makes sense
function mul!(y::LazyKet{BL}, op::LazyOperator{BL,BR}, x::LazyKet{BR}) where {BL, BR}
    T = promote_type(eltype(y), eltype(op), eltype(x))
    mul!(y, op, x, one(T), zero(T))
end
function mul!(y::LazyKet{BL}, op::LazyTensor{BL, BR}, x::LazyKet{BR}, alpha, beta) where {BL, BR}
    iszero(beta) || throw("Error: cannot perform muladd operation on LazyKets since addition is not implemented. Convert them to dense kets using Ket(x) in order to perform muladd operations.")

    iszero(alpha) && (_zero_op_mul!(y.kets[1].data, beta); return result)

    missing_index_allowed = samebases(op.basis_l, op.basis_r)
    (length(y.basis) == length(x.basis)) || throw(IncompatibleBases())

    for i in 1:length(y.kets)
        if i ∈ op.indices
            mul!(y.kets[i], op.operators[i], x.kets[i])
        else
            missing_index_allowed || throw("Can't multiply a LazyOperator with a Ket when there's missing indices and the bases are different.
                A missing index is equivalent to applying an identity operator, but that's not possible when mapping from one basis to another!")
            
            copyto!(y.kets[i].data, x.kets[i].data)
        end
    end

    rmul!(y.kets[1].data, op.factor * alpha)
    return y
end
