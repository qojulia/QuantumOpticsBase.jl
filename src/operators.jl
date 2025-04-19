import Base: ==, +, -, *, /, ^, length, one, exp, conj, conj!, transpose
import LinearAlgebra: tr, ishermitian
import SparseArrays: sparse
import QuantumInterface: AbstractOperator, AbstractKet

"""
Abstract type for operators with a data field.

This is an abstract type for operators that have a direct matrix representation
stored in their `.data` field.
"""
abstract type DataOperator{BL,BR} <: AbstractOperator end


# Common error messages
using QuantumInterface: arithmetic_binary_error, arithmetic_unary_error, addnumbererror


"""
    embed(basis1[, basis2], indices::Vector, op::AbstractOperator)

Embed operator acting on a joint Hilbert space where missing indices are filled up with identity operators.
"""
function embed(bl::Basis, br::Basis, indices, op::T) where T<:DataOperator
    (length(bl) == length(br)) || throw(ArgumentError("Must have length(bl) == length(br) in embed"))
    N = length(bl)

    reduce(tensor, bl[indices]) == basis_l(op) || throw(IncompatibleBases())
    reduce(tensor, br[indices]) == basis_r(op) || throw(IncompatibleBases())

    index_order = [idx for idx in 1:N if idx ∉ indices]
    all_operators = AbstractOperator[identityoperator(T, eltype(op), bl[i], br[i]) for i in index_order]

    for idx in indices
        pushfirst!(index_order, idx)
    end
    push!(all_operators, op)

    check_indices(N, indices)

    # Create the operator.
    permuted_op = tensor(all_operators...)

    # Reorient the matrix to act in the correctly ordered basis.
    # Get the dimensions necessary for index permuting.
    dims_l = shape(bl)
    dims_r = shape(br)

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
    unpermuted_op = constructor.wrapper(bl, br, M)

    return unpermuted_op
end

function embed(bl::Basis, br::Basis, index::Integer, op::DataOperator)
    N = length(bl)

    # Check stuff
    @assert length(br) == N
    bl[index] == basis_l(op) || throw(IncompatibleBases())
    br[index] == basis_r(op) || throw(IncompatibleBases())
    check_indices(N, index)

    # Build data
    Tnum = eltype(op)
    data = similar(sparse(op.data),1,1)
    data[1] = one(Tnum)
    i = N
    while i > 0
        if i == index
            data = kron(data, op.data)
        else
            id = SparseMatrixCSC{Tnum}(I, dimension(bl[i]), dimension(br[i]))
            data = kron(data, id)
        end
        i -= 1
    end

    return Operator(bl, br, data)
end

"""
    expect(op, state)

Expectation value of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
function expect(op::AbstractOperator, state::Ket)
    check_multiplicable(op,op); check_multiplicable(op,state)
    dot(state.data, (op * state).data)
end

# TODO upstream this one
# expect(op::AbstractOperator{B,B}, state::AbstractKet{B}) where B = norm(op * state) ^ 2

function expect(indices, op::AbstractOperator, state::Ket)
    N = length(basis(state))
    indices_ = complement(N, indices)
    expect(op, ptrace(state, indices_))
end

expect(index::Integer, op::AbstractOperator, state::Ket) = expect([index], op, state)

"""
    variance(op, state)

Variance of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
function variance(op::AbstractOperator, state::Ket)
    check_multiplicable(op,op); check_multiplicable(op,state)
    x = op*state
    state.data'*(op*x).data - (state.data'*x.data)^2
end

function variance(indices, op::AbstractOperator, state::Ket)
    N = length(basis(state))
    indices_ = complement(N, indices)
    variance(op, ptrace(state, indices_))
end

variance(index::Integer, op::AbstractOperator, state::Ket) = variance([index], op, state)

# Helper functions to check validity of arguments
function check_ptrace_arguments(a::AbstractOperator, indices)
    rank = length(basis_l(a))
    if rank != length(basis_r(a))
        throw(ArgumentError("Partial trace can only be applied onto operators wich have the same number of subsystems in the left basis and right basis."))
    end
    if rank < 2
        throw(ArgumentError("Partial trace can only be applied to operators over at least two subsystems."))
    end
    if rank == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    check_indices(length(basis_l(a)), indices)
    for i=indices
        if shape(basis_l(a))[i] != shape(basis_r(a))[i]
            throw(ArgumentError("Partial trace can only be applied onto subsystems that have the same left and right dimension."))
        end
    end
end
function check_ptrace_arguments(a::StateVector, indices)
    if length(basis(a)) == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    check_indices(length(basis(a)), indices)
end
