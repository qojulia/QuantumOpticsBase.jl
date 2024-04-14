struct SortedVector{T, OT} <: AbstractVector{T}
    sortedvector::Vector{T}
    order::OT
    function SortedVector(occ::AbstractVector{T}, order::OT=Base.Order.Forward) where {T, OT}
        if issorted(occ, order=order)
            new{T, OT}(occ, order)
        else
            new{T, OT}(sort(occ, order=order), order)
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
    SortedVector(union(sv1.sortedvector, (occ.sortedvector for occ in svs)...), sv1.order)

# Special methods for fast operator construction
function state_index(sv::SortedVector{T}, occ::T) where {T}
    ret = searchsortedfirst(sv.sortedvector, occ, order = sv.order)
    ret == length(sv) + 1 && return nothing
    return sv.sortedvector[ret] == occ ? ret : nothing
end
state_index(occupations::AbstractVector{T}, state::T) where {T} = findfirst(==(state), occupations)
state_index(occupations::AbstractVector{T}, state::Base.RefValue{T}) where {T} = state_index(occupations, state[])
state_index(occupations::AbstractVector{T}, state::Any) where {T} = state_index(occupations, convert(T, state))

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
    function ManyBodyBasis{B,O}(onebodybasis::B, occupations::O) where {B,O<:AbstractVector}
        h = hash(hash.(occupations))
        new{B,O,typeof(h)}(length(occupations), onebodybasis, occupations, h)
    end
end
ManyBodyBasis(onebodybasis::B, occupations::O) where {B,O} = ManyBodyBasis{B,O}(onebodybasis, occupations)
ManyBodyBasis(onebodybasis::B, occupations::Vector{T}) where {B,T} = ManyBodyBasis(onebodybasis, SortedVector(occupations))

allocate_buffer(occ) = similar(occ)
allocate_buffer(mb::ManyBodyBasis) = allocate_buffer(first(mb.occupations))

"""
    fermionstates([T, ]Nmodes, Nparticles)
    fermionstates([T, ]b, Nparticles)

Generate all fermionic occupation states for N-particles in M-modes.
`Nparticles` can be a vector to define a Hilbert space with variable
particle number. `T` is the type of the occupation states - default is `Vector{Int}`,
but can also be `FermionBitstring`.
"""
function fermionstates(T::Type, Nmodes::Int, Nparticles::Int)
    occ_buffer = zero(similar(T, Nmodes))
    OT = typeof(occ_buffer)
    SortedVector(_distribute_fermions(Nparticles, Nmodes, 1, occ_buffer, OT[]), Base.Reverse)
end
fermionstates(T::Type, Nmodes::Int, Nparticles::Vector{Int}) = union((fermionstates(T, Nmodes, N) for N in Nparticles)...)
fermionstates(T::Type, onebodybasis::Basis, Nparticles) = fermionstates(T, length(onebodybasis), Nparticles)
fermionstates(arg1, arg2) = fermionstates(OccupationNumbers{FermionStatistics,Int}, arg1, arg2)

"""
    bosonstates(Nmodes, Nparticles)
    bosonstates(b, Nparticles)

Generate all bosonic occupation states for N-particles in M-modes.
`Nparticles` can be a vector to define a Hilbert space with variable
particle number.
"""
bosonstates(Nmodes::Int, Nparticles::Int) = SortedVector(
    _distribute_bosons(Nparticles, Nmodes, 1,
    OccupationNumbers(BosonStatistics(), zeros(Int, Nmodes)),
    OccupationNumbers{BosonStatistics, Int}[]),
    Base.Reverse)
bosonstates(Nmodes::Int, Nparticles::Vector{Int}) = union((bosonstates(Nmodes, N) for N in Nparticles)...)
bosonstates(onebodybasis::Basis, Nparticles) = bosonstates(length(onebodybasis), Nparticles)

==(b1::ManyBodyBasis, b2::ManyBodyBasis) = b1.occupations_hash == b2.occupations_hash && b1.onebodybasis == b2.onebodybasis

"""
    basisstate([T=ComplexF64,] mb::ManyBodyBasis, occupation::Vector)

Return a ket state where the system is in the state specified by the given
occupation numbers.
"""
function basisstate(::Type{T}, mb::ManyBodyBasis, occupation::Vector) where {T}
    index = state_index(mb.occupations, occupation)
    if isa(index, Nothing)
        throw(ArgumentError("Occupation not included in many-body basis."))
    end
    basisstate(T, mb, index)
end

"""
    create([T=ComplexF64,] mb::ManyBodyBasis, index)

Creation operator for the i-th mode of the many-body basis `b`.
"""
create(::Type{T}, mb::ManyBodyBasis, index) where {T} = transition(T, mb, index, ())
create(mb::ManyBodyBasis, index) = create(ComplexF64, mb, index)

"""
    destroy([T=ComplexF64,] mb::ManyBodyBasis, index)

Annihilation operator for the i-th mode of the many-body basis `b`.
"""
destroy(::Type{T}, mb::ManyBodyBasis, index) where {T} = transition(T, mb, (), index)
destroy(mb::ManyBodyBasis, index) = destroy(ComplexF64, mb, index)

"""
    number([T=ComplexF64,] mb::ManyBodyBasis, index)

Particle number operator for the i-th mode of the many-body basis `b`.
"""
function number(::Type{T}, mb::ManyBodyBasis, index) where {T}
    diagonaloperator(mb, T[occ[index] for occ in mb.occupations])
end
number(mb::ManyBodyBasis, index) = number(ComplexF64, mb, index)

"""
    number([T=ComplexF64,] mb::ManyBodyBasis)

Total particle number operator.
"""
function number(::Type{T}, mb::ManyBodyBasis) where {T}
    diagonaloperator(mb, T[sum(occ) for occ in mb.occupations])
end
number(mb::ManyBodyBasis) = number(ComplexF64, mb)

"""
    transition([T=ComplexF64,] mb::ManyBodyBasis, to, from)

Operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|`` transferring particles between modes.

Note that `to` and `from` can be collections of indices. The resulting operator in this case
will be equal to ``a^\\dagger_{to_1} a^\\dagger_{to_2} \\ldots a_{from_2} a_{from_1}``.
"""
function transition(::Type{T}, mb::ManyBodyBasis, to, from) where {T}
    Is = Int[]
    Js = Int[]
    Vs = T[]
    buffer = allocate_buffer(mb)
    # <{m}_j| at_to a_from |{m}_i>
    for (i, occ_i) in enumerate(mb.occupations)
        C = state_transition!(buffer, occ_i, to, from)
        C === nothing && continue
        j = state_index(mb.occupations, buffer)
        j === nothing && continue
        push!(Is, j)
        push!(Js, i)
        push!(Vs, C)
    end
    return SparseOperator(mb, sparse(Is, Js, Vs, length(mb), length(mb)))
end
transition(mb::ManyBodyBasis, to, from) = transition(ComplexF64, mb, to, from)

# Calculate many-Body operator from one-body operator
"""
    manybodyoperator(mb::ManyBodyBasis, op)

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
function manybodyoperator(mb::ManyBodyBasis, op)
    @assert op.basis_l == op.basis_r
    if op.basis_l == mb.onebodybasis
        result = manybodyoperator_1(mb, op)
    elseif op.basis_l == mb.onebodybasis ⊗ mb.onebodybasis
        result = manybodyoperator_2(mb, op)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis."))
    end
    result
end

function manybodyoperator_1(mb::ManyBodyBasis, op::Operator)
    S = length(mb.onebodybasis)
    result = DenseOperator(mb)
    buffer = allocate_buffer(mb)
    @inbounds for j = 1:S, i = 1:S
        value = op.data[i, j]
        iszero(value) && continue
        for (m, occ) in enumerate(mb.occupations)
            C = state_transition!(buffer, occ, j, i)
            C === nothing && continue
            n = state_index(mb.occupations, buffer)
            n === nothing && continue
            result.data[m, n] += C * value
        end
    end
    return result
end
manybodyoperator_1(mb::ManyBodyBasis, op::AdjointOperator) = dagger(manybodyoperator_1(mb, dagger(op)))

function manybodyoperator_1(mb::ManyBodyBasis, op::SparseOpPureType)
    N = length(mb)
    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    buffer = allocate_buffer(mb)
    @inbounds for (row, column, value) in zip(findnz(op.data)...)
        for (m, occ) in enumerate(mb.occupations)
            C = state_transition!(buffer, occ, column, row)
            C === nothing && continue
            n = state_index(mb.occupations, buffer)
            n === nothing && continue
            push!(Is, m)
            push!(Js, n)
            push!(Vs, C * value)
        end
    end
    return SparseOperator(mb, sparse(Is, Js, Vs, N, N))
end

function manybodyoperator_2(mb::ManyBodyBasis, op::Operator)
    S = length(mb.onebodybasis)
    result = DenseOperator(mb)
    op_data = reshape(op.data, S, S, S, S)
    buffer = allocate_buffer(mb)
    @inbounds for l = 1:S, k = 1:S, j = 1:S, i = 1:S
        value = op_data[i, j, k, l]
        iszero(value) && continue
        for (m, occ) in enumerate(mb.occupations)
            C = state_transition!(buffer, occ, (k, l), (i, j))
            C === nothing && continue
            n = state_index(mb.occupations, buffer)
            n === nothing && continue
            result.data[m, n] += C * value
        end
    end
    return result
end

function manybodyoperator_2(mb::ManyBodyBasis, op::SparseOpType)
    N = length(mb)
    S = length(mb.onebodybasis)
    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    buffer = allocate_buffer(mb)
    @inbounds for (row, column, value) in zip(findnz(op.data)...)
        for (m, occ) in enumerate(mb.occupations)
            index = Tuple(CartesianIndices((S, S, S, S))[(column-1)*S^2+row])
            C = state_transition!(buffer, occ, index[3:4], index[1:2])
            C === nothing && continue
            n = state_index(mb.occupations, buffer)
            n === nothing && continue
            push!(Is, m)
            push!(Js, n)
            push!(Vs, C * value)
        end
    end
    return SparseOperator(mb, sparse(Is, Js, Vs, N, N))
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

matrix_element(state::Ket, m, n) = conj(state.data[m]) * state.data[n]
matrix_element(state::Operator, m, n) = state.data[n, m]
function onebodyexpect_1(op::Operator, state)
    mb = basis(state)
    occupations = mb.occupations
    S = length(mb.onebodybasis)
    buffer = allocate_buffer(mb)
    result = complex(0.0)
    for i = 1:S, j = 1:S
        value = op.data[i, j]
        iszero(value) && continue
        for (m, occ) in enumerate(occupations)
            C = state_transition!(buffer, occ, j, i)
            C === nothing && continue
            n = state_index(occupations, buffer)
            n === nothing && continue
            result += C * value * matrix_element(state, m, n)
        end
    end
    result
end

function onebodyexpect_1(op::SparseOpPureType, state)
    mb = basis(state)
    occupations = mb.occupations
    buffer = allocate_buffer(mb)
    result = complex(0.0)
    @inbounds for (row, column, value) in zip(findnz(op.data)...)
        for (m, occ) in enumerate(occupations)
            C = state_transition!(buffer, occ, column, row)
            C === nothing && continue
            n = state_index(occupations, buffer)
            n === nothing && continue
            result += C * value * matrix_element(state, m, n)
        end
    end
    result
end

# Occupations as Vector{Int}
struct OccupationNumbers{StatisticsT,T} <: AbstractVector{T}
    statistics::StatisticsT
    occupations::Vector{T}
end
Base.size(on::OccupationNumbers) = size(on.occupations)
Base.@propagate_inbounds Base.getindex(on::OccupationNumbers, v...) = getindex(on.occupations, v...)
Base.@propagate_inbounds Base.setindex!(on::OccupationNumbers, value, v...) =
    setindex!(on.occupations, value, v...)
Base.IndexStyle(::Type{<:OccupationNumbers}) = Base.IndexLinear()
Base.similar(occ::OccupationNumbers, ::Type{T}, dims::Dims) where {T} =
    OccupationNumbers(occ.statistics, similar(occ.occupations, T, dims))
Base.similar(::Type{OccupationNumbers{StatisticsT,T}}, dims::Dims) where {StatisticsT,T} =
    OccupationNumbers(StatisticsT(), similar(Vector{T}, dims))
Base.convert(::Type{OccupationNumbers{StatisticsT,T}}, occ::AbstractVector) where {StatisticsT,T} =
    OccupationNumbers(StatisticsT(), convert(Vector{T}, occ))

struct FermionStatistics end
struct BosonStatistics end

Base.@propagate_inbounds function state_transition!(buffer, occ::OccupationNumbers{BosonStatistics},
        at_indices, a_indices)
    factor_sq = 1
    copyto!(buffer, occ)
    for i in a_indices
        factor_sq *= buffer[i]
        factor_sq == 0 && return nothing
        buffer[i] -= 1
    end
    for i in at_indices
        buffer[i] += 1
        factor_sq *= buffer[i]
    end
    return √factor_sq
end

Base.@propagate_inbounds function state_transition!(buffer, occ::OccupationNumbers{FermionStatistics},
        at_indices, a_indices)
    factor = 1
    copyto!(buffer, occ)
    for i in a_indices
        buffer[i] == 0 && return nothing
        buffer[i] = 0
        ncomm = count(==(1), @view buffer[1:i-1])
        isodd(ncomm) && (factor *= -1)
    end
    for i in at_indices
        buffer[i] == 1 && return nothing
        buffer[i] = 1
        ncomm = count(==(1), @view buffer[1:i-1])
        isodd(ncomm) && (factor *= -1)
    end
    return factor
end

"""
    FermionBitstring(bits, n)

Fermionic occupation state represented as a bitstring. The bitstring `bits` is represented
as an unsigned integer with the `n` least significant bits representing the occupation
state. The bitstring is assumed to be in the canonical form where the bits are ordered
from left to right with the most significant bit being the first mode.

---

    FermionBitstring(bits, n)

Create a `FermionBitstring` from an unsigned integer `bits` with the `n` least significant
bits representing the occupation state. All remaining bits are set to zero.
"""
struct FermionBitstring{T<:Unsigned}
    bits::T
    n::Int
    function FermionBitstring{T}(bits, n) where T<:Unsigned
        n > sizeof(T) * 8 && throw(ArgumentError("n must be less than $(sizeof(T) * 8)"))
        new{T}(bits, n)
    end
    function FermionBitstring(bits::T, n::Int) where T<:Unsigned
        n > sizeof(T) * 8 && throw(ArgumentError("n must be less than $(sizeof(T) * 8)"))
        nrest = sizeof(bits) * 8 - n
        FermionBitstring{T}(UInt((bits << nrest) >>> nrest), n)
    end
end
FermionBitstring(bits::Integer, n::Int) = FermionBitstring(unsigned(bits), n)

Base.zero(fb::FermionBitstring) = FermionBitstring(zero(fb.bits), fb.n)
Base.similar(::Type{FermionBitstring{T}}, n::Int) where {T} = FermionBitstring(zero(T), n)
function Base.similar(::Type{FermionBitstring}, n::Int)
    for type in (UInt8, UInt16, UInt32, UInt64, UInt128)
        n ≤ sizeof(type) * 8 && return FermionBitstring(zero(type), n)
    end
    throw(ArgumentError("n must be less than 128; got $n"))
end
function Base.convert(::Type{FermionBitstring{T}}, v::AbstractVector) where {T}
    n = length(v)
    n > sizeof(T) * 8 && throw(ArgumentError("n must be less than $(sizeof(T) * 8)"))
    bits = zero(T)
    for i in 1:n
        v[i] in (0, 1) || throw(ArgumentError("Occupations must be 0 or 1"))
        v[i] == 1 && (bits |= one(T) << (n - i))
    end
    FermionBitstring{T}(bits, n)
end
Base.copy(fb::FermionBitstring) = fb
allocate_buffer(fb::FermionBitstring) = Ref(fb)
@inline Base.:(==)(fb1::FermionBitstring, fb2::FermionBitstring) =
    fb1.bits == fb2.bits && fb1.n == fb2.n
@inline Base.isless(fb1::FermionBitstring, fb2::FermionBitstring) =
    fb1.bits < fb2.bits || fb1.bits == fb2.bits && fb1.n < fb2.n
Base.show(io::IO, fb::FermionBitstring{T}) where {T} =
    print(io, "FermionBitstring{$T}(0b", bitstring(fb.bits)[end-fb.n+1:end], ", ", fb.n, ")")

Base.@propagate_inbounds function Base.getindex(fb::FermionBitstring, i::Int)
    @boundscheck i in 1:fb.n || throw(BoundsError(fb, i))
    Bool((fb.bits >>> (fb.n - i)) & 1)
end
Base.@propagate_inbounds function write_bit(fb::FermionBitstring, i::Int, value::Bool)
    @boundscheck i in 1:fb.n || throw(BoundsError(fb, i))
    offset = fb.n - i
    value ? FermionBitstring(fb.bits | (one(fb.bits) << offset), fb.n) :
            FermionBitstring(fb.bits & ~(one(fb.bits) << offset), fb.n)
end
Base.@propagate_inbounds function write_bit(v::AbstractVector, i::Int, value::Bool)
    setindex!(v, value, i)
    return v
end
Base.@propagate_inbounds function state_transition!(buffer, occ::FermionBitstring, at_indices, a_indices)
    factor = 1
    for i in a_indices
        occ[i] || return nothing
        occ = write_bit(occ, i, false)
        ncomm = count_ones(occ.bits >>> (occ.n - i + 1))
        isodd(ncomm) && (factor *= -1)
    end
    for i in at_indices
        occ[i] && return nothing
        occ = write_bit(occ, i, true)
        ncomm = count_ones(occ.bits >>> (occ.n - i + 1))
        isodd(ncomm) && (factor *= -1)
    end
    buffer[] = occ
    return factor
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
    if Nparticles == 0
        push!(results, copy(occupations))
        return results
    end
    for new_index in index:Nmodes - Nparticles + 1
        occupations = write_bit(occupations, new_index, true)
        _distribute_fermions(Nparticles - 1, Nmodes, new_index + 1, occupations, results)
        occupations = write_bit(occupations, new_index, false)
    end
    return results
end
