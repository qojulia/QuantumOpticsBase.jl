import Base: ==, +, -, *, /, length, copy, eltype
import LinearAlgebra: norm, normalize, normalize!
import QuantumInterface: StateVector, AbstractKet, AbstractBra

"""
    Bra(b::Basis[, data])

Bra state defined by coefficients in respect to the basis.
"""
mutable struct Bra{B,T} <: AbstractBra{B,T}
    basis::B
    data::T
    function Bra{B,T}(b::B, data::T) where {B,T}
        (length(b)==length(data)) || throw(DimensionMismatch("Tried to assign data of length $(length(data)) to Hilbert space of size $(length(b))"))
        new(b, data)
    end
end

"""
    Ket(b::Basis[, data])

Ket state defined by coefficients in respect to the given basis.
"""
mutable struct Ket{B,T} <: AbstractKet{B,T}
    basis::B
    data::T
    function Ket{B,T}(b::B, data::T) where {B,T}
        (length(b)==length(data)) || throw(DimensionMismatch("Tried to assign data of length $(length(data)) to Hilbert space of size $(length(b))"))
        new(b, data)
    end
end

Base.zero(x::Bra) = Bra(x.basis, zero(x.data))
Base.zero(x::Ket) = Ket(x.basis, zero(x.data))
eltype(::Type{K}) where {K <: Ket{B,V}} where {B,V} = eltype(V)
eltype(::Type{K}) where {K <: Bra{B,V}} where {B,V} = eltype(V)

Bra{B}(b::B, data::T) where {B,T} = Bra{B,T}(b, data)
Ket{B}(b::B, data::T) where {B,T} = Ket{B,T}(b, data)

Bra(b::B, data::T) where {B,T} = Bra{B,T}(b, data)
Ket(b::B, data::T) where {B,T} = Ket{B,T}(b, data)

Bra{B}(::Type{T}, b::B) where {T,B} = Bra{B}(b, zeros(T, length(b)))
Ket{B}(::Type{T}, b::B) where {T,B} = Ket{B}(b, zeros(T, length(b)))
Bra(::Type{T}, b::Basis) where T = Bra(b, zeros(T, length(b)))
Ket(::Type{T}, b::Basis) where T = Ket(b, zeros(T, length(b)))

Bra{B}(b::B) where B = Bra{B}(ComplexF64, b)
Ket{B}(b::B) where B = Ket{B}(ComplexF64, b)
Bra(b::Basis) = Bra(ComplexF64, b)
Ket(b::Basis) = Ket(ComplexF64, b)

==(x::Ket{B}, y::Ket{B}) where {B} = (samebases(x, y) && x.data==y.data)
==(x::Bra{B}, y::Bra{B}) where {B} = (samebases(x, y) && x.data==y.data)
==(x::Ket, y::Ket) = false
==(x::Bra, y::Bra) = false

Base.isapprox(x::Ket{B}, y::Ket{B}; kwargs...) where {B} = (samebases(x, y) && isapprox(x.data,y.data;kwargs...))
Base.isapprox(x::Bra{B}, y::Bra{B}; kwargs...) where {B} = (samebases(x, y) && isapprox(x.data,y.data;kwargs...))
Base.isapprox(x::Ket, y::Ket; kwargs...) = false
Base.isapprox(x::Bra, y::Bra; kwargs...) = false

# Arithmetic operations
+(a::Ket{B}, b::Ket{B}) where {B} = Ket(a.basis, a.data+b.data)
+(a::Bra{B}, b::Bra{B}) where {B} = Bra(a.basis, a.data+b.data)
+(a::Ket, b::Ket) = throw(IncompatibleBases())
+(a::Bra, b::Bra) = throw(IncompatibleBases())

-(a::Ket{B}, b::Ket{B}) where {B} = Ket(a.basis, a.data-b.data)
-(a::Bra{B}, b::Bra{B}) where {B} = Bra(a.basis, a.data-b.data)
-(a::Ket, b::Ket) = throw(IncompatibleBases())
-(a::Bra, b::Bra) = throw(IncompatibleBases())

*(a::Bra{B}, b::Ket{B}) where {B} = transpose(a.data)*b.data
*(a::Bra, b::Ket) = throw(IncompatibleBases())
*(a::Number, b::Ket) = Ket(b.basis, a*b.data)
*(a::Number, b::Bra) = Bra(b.basis, a*b.data)

/(a::Ket, b::Number) = Ket(a.basis, a.data ./ b)
/(a::Bra, b::Number) = Bra(a.basis, a.data ./ b)


"""
    dagger(x)

Hermitian conjugate.
"""
dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))

"""
    tensor(x::Ket, y::Ket, z::Ket...)

Tensor product ``|x⟩⊗|y⟩⊗|z⟩⊗…`` of the given states.
"""
tensor(a::Ket, b::Ket) = Ket(tensor(a.basis, b.basis), kron(b.data, a.data))
tensor(a::Bra, b::Bra) = Bra(tensor(a.basis, b.basis), kron(b.data, a.data))
tensor(states::Ket...) = reduce(tensor, states)
tensor(states::Bra...) = reduce(tensor, states)

function permutesystems(state::T, perm) where T<:Ket
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, state.basis.shape...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    Ket(permutesystems(state.basis, perm), data)
end
function permutesystems(state::T, perm) where T<:Bra
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, state.basis.shape...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    Bra(permutesystems(state.basis, perm), data)
end

# Creation of basis states.
"""
    basisstate([T=ComplexF64, ]b, index)

Basis vector specified by `index` as ket state.

For a composite system `index` can be a vector which then creates a tensor
product state ``|i_1⟩⊗|i_2⟩⊗…⊗|i_n⟩`` of the corresponding basis states.
"""
function basisstate(::Type{T}, b::Basis, indices) where T
    @assert length(b.shape) == length(indices)
    x = zeros(T, length(b))
    x[LinearIndices(tuple(b.shape...))[indices...]] = one(T)
    Ket(b, x)
end
function basisstate(::Type{T}, b::Basis, index::Integer) where T
    data = zeros(T, length(b))
    data[index] = one(T)
    Ket(b, data)
end
basisstate(b::Basis, indices) = basisstate(ComplexF64, b, indices)

"""
    sparsebasisstate([T=ComplexF64, ]b, index)

Sparse version of [`basisstate`](@ref).
"""
function sparsebasisstate(::Type{T}, b::Basis, indices) where T
    @assert length(b.shape) == length(indices)
    x = spzeros(T, length(b))
    x[LinearIndices(tuple(b.shape...))[indices...]] = one(T)
    Ket(b, x)
end
function sparsebasisstate(::Type{T}, b::Basis, index::Integer) where T
    data = spzeros(T, length(b))
    data[index] = one(T)
    Ket(b, data)
end
sparsebasisstate(b::Basis, indices) = sparsebasisstate(ComplexF64, b, indices)

SparseArrays.sparse(x::Ket) = Ket(x.basis,sparse(x.data))
SparseArrays.sparse(x::Bra) = Bra(x.basis,sparse(x.data))

# Helper functions to check validity of arguments
function check_multiplicable(a::Bra, b::Ket)
    if a.basis != b.basis
        throw(IncompatibleBases())
    end
end

samebases(a::Ket{B}, b::Ket{B}) where {B} = samebases(a.basis, b.basis)::Bool
samebases(a::Bra{B}, b::Bra{B}) where {B} = samebases(a.basis, b.basis)::Bool

# Custom broadcasting style
abstract type StateVectorStyle{B} <: Broadcast.BroadcastStyle end
struct KetStyle{B} <: StateVectorStyle{B} end
struct BraStyle{B} <: StateVectorStyle{B} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:Ket{B}}) where {B} = KetStyle{B}()
Broadcast.BroadcastStyle(::Type{<:Bra{B}}) where {B} = BraStyle{B}()
Broadcast.BroadcastStyle(::KetStyle{B1}, ::KetStyle{B2}) where {B1,B2} = throw(IncompatibleBases())
Broadcast.BroadcastStyle(::BraStyle{B1}, ::BraStyle{B2}) where {B1,B2} = throw(IncompatibleBases())

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:KetStyle{B},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    b = find_basis(bcf)
    return Ket{B}(b, copy(bc_))
end
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:BraStyle{B},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    b = find_basis(bcf)
    return Bra{B}(b, copy(bc_))
end
find_basis(bc::Broadcast.Broadcasted) = find_basis(bc.args)
find_basis(args::Tuple) = find_basis(find_basis(args[1]), Base.tail(args))
find_basis(x) = x
find_basis(a::StateVector, rest) = a.basis
find_basis(::Any, rest) = find_basis(rest)

const BasicMathFunc = Union{typeof(+),typeof(-),typeof(*)}
function Broadcasted_restrict_f(f::BasicMathFunc, args::NTuple{N,<:T}, axes) where {T<:StateVector,N}
    args_ = Tuple(a.data for a=args)
    return Broadcast.Broadcasted(f, args_, axes)
end
function Broadcasted_restrict_f(f, args::Tuple, axes)
    error("Cannot broadcast function `$f` on $(typeof(args))")
end
function Broadcasted_restrict_f(f::BasicMathFunc, args::Tuple{}, axes) # Defined to avoid method ambiguities
    error("Cannot broadcast function `$f` on an empty set of arguments")
end

# In-place broadcasting for Kets
@inline function Base.copyto!(dest::Ket{B}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:KetStyle{B},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && isa(bc.args, Tuple{<:Ket{B}}) # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    # Get the underlying data fields of kets and broadcast them as arrays
    bcf = Broadcast.flatten(bc)
    args_ = Tuple(a.data for a=bcf.args)
    bc_ = Broadcast.Broadcasted(bcf.f, args_, axes(bcf))
    copyto!(dest.data, bc_)
    return dest
end
@inline Base.copyto!(dest::Ket{B1}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B1,B2,Style<:KetStyle{B2},Axes,F,Args} =
    throw(IncompatibleBases())

# In-place broadcasting for Bras
@inline function Base.copyto!(dest::Bra{B}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:BraStyle{B},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && isa(bc.args, Tuple{<:Bra{B}}) # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    # Get the underlying data fields of bras and broadcast them as arrays
    bcf = Broadcast.flatten(bc)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    copyto!(dest.data, bc_)
    return dest
end
@inline Base.copyto!(dest::Bra{B1}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B1,B2,Style<:BraStyle{B2},Axes,F,Args} =
    throw(IncompatibleBases())

@inline Base.copyto!(A::T,B::T) where T<:Union{Ket, Bra} = (copyto!(A.data,B.data); A) # Can not use T<:QuantumInterface.StateVector, because StateVector does not imply the existence of a data property
