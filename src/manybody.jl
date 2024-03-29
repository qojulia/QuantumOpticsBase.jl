struct SortedVector{T, OT} <: AbstractVector{T}
    sortedvector::Vector{T}
    ord::OT
    function SortedVector(occ::AbstractVector{T}, ord::OT=Base.Order.Forward) where {T, OT}
        if issorted(occ, order=ord)
            new{T, OT}(occ, ord)
        else
            new{T, OT}(sort(occ, order=ord), ord)
        end
    end
end
Base.:(==)(sv1::SortedVector, sv2::SortedVector) = sv1.sortedvector == sv2.sortedvector
Base.size(sv::SortedVector) = (length(sv.sortedvector),)
Base.@propagate_inbounds function Base.getindex(sv::SortedVector, i::Int)
    @boundscheck !checkbounds(Bool, sv.sortedvector, i) && throw(BoundsError(sv, i))
    return sv.sortedvector[i]
end
Base.union(sv1::SortedVector{T}, svs::SortedVector{T}...) where {T} =
    SortedVector(union(sv1.sortedvector, (occ.sortedvector for occ in svs)...))

# Special methods for fast operator construction
allocate_buffer(sv::AbstractVector) = similar(first(sv))
function state_index(sv::SortedVector, state)
    ret = searchsortedfirst(sv.sortedvector, state, order = sv.ord)
    ret == length(sv) + 1 && return nothing
    return sv.sortedvector[ret] == state ? ret : nothing
end
state_index(sv::AbstractVector, state) = findfirst(==(state), sv)

"""
    ManyBodyBasis(b, occupations)

Basis for a many body system.

The basis has to know the associated one-body basis `b` and which occupation states
should be included. The occupations_hash is used to speed up checking if two
many-body bases are equal.
"""
struct ManyBodyBasis{B,O,UT} <: Basis
    shape::Int
    onebodybasis::B
    occupations::O
    occupations_hash::UT
    function ManyBodyBasis{B,O}(onebodybasis::B, occupations::O) where {B,O<:AbstractVector{Vector{Int}}}
        h = hash(hash.(occupations))
        new{B,O,typeof(h)}(length(occupations), onebodybasis, occupations, h)
    end
end
ManyBodyBasis(onebodybasis::B, occupations::O) where {B,O} = ManyBodyBasis{B,O}(onebodybasis, occupations)
ManyBodyBasis(onebodybasis::B, occupations::Vector{T}) where {B,T} = ManyBodyBasis(onebodybasis, SortedVector(occupations))

"""
    fermionstates(Nmodes, Nparticles)
    fermionstates(b, Nparticles)

Generate all fermionic occupation states for N-particles in M-modes.
`Nparticles` can be a vector to define a Hilbert space with variable
particle number.
"""
fermionstates(Nmodes::Int, Nparticles::Int) = SortedVector(_distribute_fermions(Nparticles, Nmodes, 1, zeros(Int, Nmodes), Vector{Int}[]), Base.Reverse)
fermionstates(Nmodes::Int, Nparticles::Vector{Int}) = union((fermionstates(Nmodes, N) for N in Nparticles)...)
fermionstates(onebodybasis::Basis, Nparticles) = fermionstates(length(onebodybasis), Nparticles)

"""
    bosonstates(Nmodes, Nparticles)
    bosonstates(b, Nparticles)

Generate all bosonic occupation states for N-particles in M-modes.
`Nparticles` can be a vector to define a Hilbert space with variable
particle number.
"""
bosonstates(Nmodes::Int, Nparticles::Int) = SortedVector(_distribute_bosons(Nparticles, Nmodes, 1, zeros(Int, Nmodes), Vector{Int}[]), Base.Reverse)
bosonstates(Nmodes::Int, Nparticles::Vector{Int}) = union((bosonstates(Nmodes, N) for N in Nparticles)...)
bosonstates(onebodybasis::Basis, Nparticles) = bosonstates(length(onebodybasis), Nparticles)

==(b1::ManyBodyBasis, b2::ManyBodyBasis) = b1.occupations_hash == b2.occupations_hash && b1.onebodybasis == b2.onebodybasis

"""
    basisstate([T=ComplexF64,] b::ManyBodyBasis, occupation::Vector)

Return a ket state where the system is in the state specified by the given
occupation numbers.
"""
function basisstate(::Type{T}, basis::ManyBodyBasis, occupation::Vector) where {T}
    index = state_index(basis.occupations, occupation)
    if isa(index, Nothing)
        throw(ArgumentError("Occupation not included in many-body basis."))
    end
    basisstate(T, basis, index)
end

"""
    create([T=ComplexF64,] b::ManyBodyBasis, index)

Creation operator for the i-th mode of the many-body basis `b`.
"""
function create(::Type{T}, b::ManyBodyBasis, index) where {T}
    Is = Int[]
    Js = Int[]
    Vs = T[]
    # <{m}_i| at |{m}_j>
    buffer = allocate_buffer(b.occupations)
    for (i, occ_i) in enumerate(b.occupations)
        if occ_i[index] == 0
            continue
        end
        copyto!(buffer, occ_i)
        buffer[index] -= 1
        j = state_index(b.occupations, buffer)
        j === nothing && continue
        push!(Is, i)
        push!(Js, j)
        push!(Vs, sqrt(occ_i[index]))
    end
    return SparseOperator(b, sparse(Is, Js, Vs, length(b), length(b)))
end
create(b::ManyBodyBasis, index) = create(ComplexF64, b, index)

"""
    destroy([T=ComplexF64,] b::ManyBodyBasis, index)

Annihilation operator for the i-th mode of the many-body basis `b`.
"""
function destroy(::Type{T}, b::ManyBodyBasis, index) where {T}
    Is = Int[]
    Js = Int[]
    Vs = T[]
    buffer = allocate_buffer(b.occupations)
    # <{m}_j| a |{m}_i>
    for (i, occ_i) in enumerate(b.occupations)
        if occ_i[index] == 0
            continue
        end
        copyto!(buffer, occ_i)
        buffer[index] -= 1
        j = state_index(b.occupations, buffer)
        j === nothing && continue
        push!(Is, j)
        push!(Js, i)
        push!(Vs, sqrt(occ_i[index]))
    end
    return SparseOperator(b, sparse(Is, Js, Vs, length(b), length(b)))
end
destroy(b::ManyBodyBasis, index) = destroy(ComplexF64, b, index)

"""
    number([T=ComplexF64,] b::ManyBodyBasis, index)

Particle number operator for the i-th mode of the many-body basis `b`.
"""
function number(::Type{T}, b::ManyBodyBasis, index) where {T}
    diagonaloperator(b, T[occ[index] for occ in b.occupations])
end
number(b::ManyBodyBasis, index) = number(ComplexF64, b, index)

"""
    number([T=ComplexF64,] b::ManyBodyBasis)

Total particle number operator.
"""
function number(::Type{T}, b::ManyBodyBasis) where {T}
    diagonaloperator(b, T[sum(occ) for occ in b.occupations])
end
number(b::ManyBodyBasis) = number(ComplexF64, b)

"""
    transition([T=ComplexF64,] b::ManyBodyBasis, to, from)

Operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|`` transferring particles between modes.

Note that `to` and `from` can be collections of indices. The resulting operator in this case
will be equal to ``a^\\dagger_{to_1} a^\\dagger_{to_2} \\ldots a_{from_2} a_{from_1}``.
"""
function transition(::Type{T}, b::ManyBodyBasis, to, from) where {T}
    Is = Int[]
    Js = Int[]
    Vs = T[]
    buffer = allocate_buffer(b.occupations)
    # <{m}_j| at_to a_from |{m}_i>
    for (i, occ_i) in enumerate(b.occupations)
        C = state_transition!(buffer, occ_i, from, to)
        C === nothing && continue
        j = state_index(b.occupations, buffer)
        j === nothing && continue
        push!(Is, j)
        push!(Js, i)
        push!(Vs, C)
    end
    return SparseOperator(b, sparse(Is, Js, Vs, length(b), length(b)))
end
transition(b::ManyBodyBasis, to, from) = transition(ComplexF64, b, to, from)

# Calculate many-Body operator from one-body operator
"""
    manybodyoperator(b::ManyBodyBasis, op)

Create the many-body operator from the given one-body operator `op`.

The given operator can either be a one-body operator or a
two-body interaction. Higher order interactions are at the
moment not implemented.

The mathematical formalism for the one-body case is described by

```math
X = \\sum_{ij} a_i^† a_j ⟨u_i| x | u_j⟩
```

and for the interaction case by

```math
X = \\sum_{ijkl} a_i^† a_j^† a_k a_l ⟨u_i|⟨u_j| x |u_k⟩|u_l⟩
```

where ``X`` is the N-particle operator, ``x`` is the one-body operator and
``|u⟩`` are the one-body states associated to the
different modes of the N-particle basis.
"""
function manybodyoperator(basis::ManyBodyBasis, op)
    @assert op.basis_l == op.basis_r
    if op.basis_l == basis.onebodybasis
        result = manybodyoperator_1(basis, op)
    elseif op.basis_l == basis.onebodybasis ⊗ basis.onebodybasis
        result = manybodyoperator_2(basis, op)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis."))
    end
    result
end

function manybodyoperator_1(basis::ManyBodyBasis, op::Operator)
    S = length(basis.onebodybasis)
    result = DenseOperator(basis)
    buffer = allocate_buffer(basis.occupations)
    @inbounds for j = 1:S, i = 1:S
        value = op.data[i, j]
        iszero(value) && continue
        for (m, occ) in enumerate(basis.occupations)
            C = state_transition!(buffer, occ, i, j)
            C === nothing && continue
            n = state_index(basis.occupations, buffer)
            n === nothing && continue
            result.data[m, n] += C * value
        end
    end
    return result
end
manybodyoperator_1(basis::ManyBodyBasis, op::AdjointOperator) = dagger(manybodyoperator_1(basis, dagger(op)))

function manybodyoperator_1(basis::ManyBodyBasis, op::SparseOpPureType)
    N = length(basis)
    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    buffer = allocate_buffer(basis.occupations)
    @inbounds for (row, column, value) in zip(findnz(op.data)...)
        for (m, occ) in enumerate(basis.occupations)
            C = state_transition!(buffer, occ, row, column)
            C === nothing && continue
            n = state_index(basis.occupations, buffer)
            n === nothing && continue
            push!(Is, m)
            push!(Js, n)
            push!(Vs, C * value)
        end
    end
    return SparseOperator(basis, sparse(Is, Js, Vs, N, N))
end

function manybodyoperator_2(basis::ManyBodyBasis, op::Operator)
    S = length(basis.onebodybasis)
    @assert S^2 == length(op.basis_l)
    result = DenseOperator(basis)
    op_data = reshape(op.data, S, S, S, S)
    buffer = allocate_buffer(basis.occupations)
    @inbounds for l = 1:S, k = 1:S, j = 1:S, i = 1:S
        value = op_data[i, j, k, l]
        iszero(value) && continue
        for (m, occ) in enumerate(basis.occupations)
            C = state_transition!(buffer, occ, (i, j), (k, l))
            C === nothing && continue
            n = state_index(basis.occupations, buffer)
            n === nothing && continue
            result.data[m, n] += C * value
        end
    end
    return result
end

function manybodyoperator_2(basis::ManyBodyBasis, op::SparseOpType)
    N = length(basis)
    S = length(basis.onebodybasis)
    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    buffer = allocate_buffer(basis.occupations)
    @inbounds for (row, column, value) in zip(findnz(op.data)...)
        for (m, occ) in enumerate(basis.occupations)
            index = Tuple(CartesianIndices((S, S, S, S))[(column-1)*S^2+row])
            C = state_transition!(buffer, occ, index[1:2], index[3:4])
            C === nothing && continue
            n = state_index(basis.occupations, buffer)
            n === nothing && continue
            push!(Is, m)
            push!(Js, n)
            push!(Vs, C * value)
        end
    end
    return SparseOperator(basis, sparse(Is, Js, Vs, N, N))
end


# Calculate expectation value of one-body operator
"""
    onebodyexpect(op, state)

Expectation value of the one-body operator `op` in respect to the many-body `state`.
"""
function onebodyexpect(op::AbstractOperator, state::Union{Ket,AbstractOperator})
    bas = basis(state)
    @assert bas isa ManyBodyBasis
    @assert op.basis_l == op.basis_r
    if bas.onebodybasis == op.basis_l
        return onebodyexpect_1(op, state)
    elseif bas.onebodybasis ⊗ bas.onebodybasis == op.basis_l
        # Not yet implemented
        throw(ArgumentError("`onebodyexpect` is not implemented for two-body states yet"))
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis of the state."))
    end
end

onebodyexpect(op::AbstractOperator, states::Vector) = [onebodyexpect(op, state) for state = states]

get_value(state::Ket,  m, n) = conj(state.data[m]) * state.data[n]
get_value(state::Operator,  m, n) = state.data[n, m]
function onebodyexpect_1(op::Operator, state)
    b = basis(state)
    occupations = b.occupations
    S = length(b.onebodybasis)
    buffer = allocate_buffer(occupations)
    result = complex(0.0)
    for i = 1:S, j = 1:S
        value = op.data[i, j]
        iszero(value) && continue
        for (m, occ) in enumerate(occupations)
            C = state_transition!(buffer, occ, i, j)
            C === nothing && continue
            n = state_index(occupations, buffer)
            n === nothing && continue
            result += C * value * get_value(state, m, n)
        end
    end
    result
end

function onebodyexpect_1(op::SparseOpPureType, state)
    b = basis(state)
    occupations = b.occupations
    buffer = allocate_buffer(occupations)
    result = complex(0.0)
    @inbounds for (row, column, value) in zip(findnz(op.data)...)
        for (m, occ) in enumerate(occupations)
            C = state_transition!(buffer, occ, row, column)
            C === nothing && continue
            n = state_index(occupations, buffer)
            n === nothing && continue
            result += C * value * get_value(state, m, n)
        end
    end
    result
end

Base.@propagate_inbounds function state_transition!(buffer, occ_m, at_indices, a_indices)
    any(==(0), (occ_m[m] for m in at_indices)) && return nothing
    result = 1
    copyto!(buffer, occ_m)
    for i in at_indices
        result *= buffer[i]
        result == 0 && return nothing
        buffer[i] -= 1
    end
    for i in a_indices
        buffer[i] += 1
        result *= buffer[i]
    end
    return √result
end

function _distribute_bosons(Nparticles, Nmodes, index, occupations, results)
    if index == Nmodes
        occupations[index] = Nparticles
        push!(results, copy(occupations))
    else
        for n = Nparticles:-1:0
            occupations[index] = n
            _distribute_bosons(Nparticles - n, Nmodes, index + 1, occupations, results)
        end
    end
    return results
end

function _distribute_fermions(Nparticles, Nmodes, index, occupations, results)
    if (Nmodes - index) + 1 < Nparticles
        return results
    end
    if index == Nmodes
        occupations[index] = Nparticles
        push!(results, copy(occupations))
    else
        for n = min(1, Nparticles):-1:0
            occupations[index] = n
            _distribute_fermions(Nparticles - n, Nmodes, index + 1, occupations, results)
        end
    end
    return results
end
