import Base: ==, +, -, *, /, ^, length, one, exp, conj, conj!, transpose
import LinearAlgebra: tr, ishermitian
import SparseArrays: sparse
import QuantumInterface: AbstractOperator, AbstractKet

"""
Abstract type for operators with a data field.

This is an abstract type for operators that have a direct matrix representation
stored in their `.data` field.
"""
abstract type DataOperator{BL,BR} <: AbstractOperator{BL,BR} end


# Common error messages
using QuantumInterface: arithmetic_binary_error, arithmetic_unary_error, addnumbererror


"""
    embed(basis1[, basis2], indices::Vector, op::AbstractOperator)

Embed operator acting on a joint Hilbert space where missing indices are filled up with identity operators.
"""
function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
               indices, op::T) where T<:DataOperator
    N = length(basis_l.bases)
    @assert length(basis_r.bases) == N

    reduce(tensor, basis_l.bases[indices]) == op.basis_l || throw(IncompatibleBases())
    reduce(tensor, basis_r.bases[indices]) == op.basis_r || throw(IncompatibleBases())

    index_order = [idx for idx in 1:length(basis_l.bases) if idx ∉ indices]
    all_operators = AbstractOperator[identityoperator(T, eltype(op), basis_l.bases[i], basis_r.bases[i]) for i in index_order]

    for idx in indices
        pushfirst!(index_order, idx)
    end
    push!(all_operators, op)

    check_indices(N, indices)

    # Create the operator.
    permuted_op = tensor(all_operators...)

    # Reorient the matrix to act in the correctly ordered basis.
    # Get the dimensions necessary for index permuting.
    dims_l = [b.shape[1] for b in basis_l.bases]
    dims_r = [b.shape[1] for b in basis_r.bases]

    # Get the order of indices to use in the first reshape. Julia indices go in
    # reverse order.
    expand_order = index_order[end:-1:1]
    # Get the dimensions associated with those indices.
    expand_dims_l = dims_l[expand_order]
    expand_dims_r = dims_r[expand_order]

    # Prepare the permutation to the correctly ordered basis.
    perm_order_l = [indexin(idx, expand_order)[1] for idx in 1:length(dims_l)]
    perm_order_r = [indexin(idx, expand_order)[1] for idx in 1:length(dims_r)]

    # Perform the index expansion, the permutation, and the index collapse.
    M = (reshape(permuted_op.data, tuple([expand_dims_l; expand_dims_r]...)) |>
         x -> permutedims(x, [perm_order_l; perm_order_r .+ length(dims_l)]) |>
         x -> sparse(reshape(x, (prod(dims_l), prod(dims_r)))))

    # Create operator with proper data and bases
    constructor = Base.typename(T)
    unpermuted_op = constructor.wrapper(basis_l, basis_r, M)

    return unpermuted_op
end

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
                index::Integer, op::T) where T<:DataOperator

    N = length(basis_l.bases)

    # Check stuff
    @assert N==length(basis_r.bases)
    basis_l.bases[index] == op.basis_l || throw(IncompatibleBases())
    basis_r.bases[index] == op.basis_r || throw(IncompatibleBases())
    check_indices(N, index)

    # Build data
    Tnum = eltype(op)
    data = similar(sparse(op.data),1,1)
    data[1] = one(Tnum)
    i = N
    while i > 0
        if i == index
            data = kron(data, op.data)
            i -= length(index)
        else
            bl = basis_l.bases[i]
            br = basis_r.bases[i]
            id = SparseMatrixCSC{Tnum}(I, length(bl), length(br))
            data = kron(data, id)
            i -= 1
        end
    end

    return Operator(basis_l, basis_r, data)
end

"""
    expect(op, state)

Expectation value of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
expect(op::AbstractOperator{B,B}, state::Ket{B}) where B = dot(state.data, (op * state).data)

expect(op::AbstractOperator{B,B}, state::AbstractKet{B}) where B = norm(op * state) ^ 2

function expect(indices, op::AbstractOperator{B,B}, state::Ket{B2}) where {B,B2<:CompositeBasis}
    N = length(state.basis.shape)
    indices_ = complement(N, indices)
    expect(op, ptrace(state, indices_))
end

expect(index::Integer, op::AbstractOperator{B,B}, state::Ket{B2}) where {B,B2<:CompositeBasis} = expect([index], op, state)

"""
    variance(op, state)

Variance of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
function variance(op::AbstractOperator{B,B}, state::Ket{B}) where B
    x = op*state
    state.data'*(op*x).data - (state.data'*x.data)^2
end

function variance(indices, op::AbstractOperator{B,B}, state::Ket{BC}) where {B,BC<:CompositeBasis}
    N = length(state.basis.shape)
    indices_ = complement(N, indices)
    variance(op, ptrace(state, indices_))
end

variance(index::Integer, op::AbstractOperator{B,B}, state::Ket{BC}) where {B,BC<:CompositeBasis} = variance([index], op, state)

# Helper functions to check validity of arguments
function check_ptrace_arguments(a::AbstractOperator, indices)
    if !isa(a.basis_l, CompositeBasis) || !isa(a.basis_r, CompositeBasis)
        throw(ArgumentError("Partial trace can only be applied onto operators with composite bases."))
    end
    rank = length(a.basis_l.shape)
    if rank != length(a.basis_r.shape)
        throw(ArgumentError("Partial trace can only be applied onto operators wich have the same number of subsystems in the left basis and right basis."))
    end
    if rank == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    check_indices(length(a.basis_l.shape), indices)
    for i=indices
        if a.basis_l.shape[i] != a.basis_r.shape[i]
            throw(ArgumentError("Partial trace can only be applied onto subsystems that have the same left and right dimension."))
        end
    end
end
function check_ptrace_arguments(a::StateVector, indices)
    if length(basis(a).shape) == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    check_indices(length(basis(a).shape), indices)
end

multiplicable(a::AbstractOperator, b::Ket) = multiplicable(a.basis_r, b.basis)
multiplicable(a::Bra, b::AbstractOperator) = multiplicable(a.basis, b.basis_l)

