import Base: ==, +, -, *, /, length, copy, eltype
import LinearAlgebra: norm, normalize, normalize!
import QuantumInterface: StateVector, AbstractKet, AbstractBra

"""
    Bra(b::Basis[, data])

Bra state defined by coefficients in respect to the basis.
"""
mutable struct Bra{B<:Basis,T} <: AbstractBra
    basis::B
    data::T
    function Bra{B,T}(b::B, data::T) where {B,T}
        (dimension(b)==length(data)) || throw(DimensionMismatch("Tried to assign data of length $(length(data)) to Hilbert space of size $(dimension(b))"))
        new(b, data)
    end
end

"""
    Ket(b::Basis[, data])

Ket state defined by coefficients in respect to the given basis.
"""
mutable struct Ket{B<:Basis,T} <: AbstractKet
    basis::B
    data::T
    function Ket{B,T}(b::B, data::T) where {B,T}
        (dimension(b)==length(data)) || throw(DimensionMismatch("Tried to assign data of length $(length(data)) to Hilbert space of size $(dimension(b))"))
        new(b, data)
    end
end

basis(x::Bra) = x.basis
basis(x::Ket) = x.basis

Base.zero(x::Bra) = Bra(x.basis, zero(x.data))
Base.zero(x::Ket) = Ket(x.basis, zero(x.data))
eltype(::Type{K}) where {K <: Ket{B,V}} where {B,V} = eltype(V)
eltype(::Type{K}) where {K <: Bra{B,V}} where {B,V} = eltype(V)

Bra{B}(b::B, data::T) where {B,T} = Bra{B,T}(b, data)
Ket{B}(b::B, data::T) where {B,T} = Ket{B,T}(b, data)

Bra(b::B, data::T) where {B,T} = Bra{B,T}(b, data)
Ket(b::B, data::T) where {B,T} = Ket{B,T}(b, data)

Bra{B}(::Type{T}, b::B) where {T,B} = Bra{B}(b, zeros(T, dimension(b)))
Ket{B}(::Type{T}, b::B) where {T,B} = Ket{B}(b, zeros(T, dimension(b)))
Bra(::Type{T}, b::Basis) where T = Bra(b, zeros(T, dimension(b)))
Ket(::Type{T}, b::Basis) where T = Ket(b, zeros(T, dimension(b)))

Bra{B}(b::B) where B = Bra{B}(ComplexF64, b)
Ket{B}(b::B) where B = Ket{B}(ComplexF64, b)
Bra(b::Basis) = Bra(ComplexF64, b)
Ket(b::Basis) = Ket(ComplexF64, b)

==(x::Ket, y::Ket) = (addible(x,y) && x.data==y.data)
==(x::Bra, y::Bra) = (addible(x,y) && x.data==y.data)

Base.isapprox(x::Ket, y::Ket; kwargs...) = (addible(x, y) && isapprox(x.data,y.data;kwargs...))
Base.isapprox(x::Bra, y::Bra; kwargs...) = (addible(x, y) && isapprox(x.data,y.data;kwargs...))

# Arithmetic operations
+(a::Ket, b::Ket) = (check_addible(a,b); Ket(a.basis, a.data+b.data))
+(a::Bra, b::Bra) = (check_addible(a,b); Bra(a.basis, a.data+b.data))

-(a::Ket, b::Ket) = (check_addible(a,b); Ket(a.basis, a.data-b.data))
-(a::Bra, b::Bra) = (check_addible(a,b); Bra(a.basis, a.data-b.data))

*(a::Bra, b::Ket) = (check_multiplicable(a,b); transpose(a.data)*b.data)
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
    @assert length(basis(state)) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, shape(state.basis)...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    Ket(permutesystems(state.basis, perm), data)
end
function permutesystems(state::T, perm) where T<:Bra
    @assert length(basis(state)) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, shape(state.basis)...)
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
    @assert length(b) == length(indices)
    x = zeros(T, dimension(b))
    x[LinearIndices(shape(b))[indices...]] = one(T)
    Ket(b, x)
end
function basisstate(::Type{T}, b::Basis, index::Integer) where T
    data = zeros(T, dimension(b))
    data[index] = one(T)
    Ket(b, data)
end
basisstate(b::Basis, indices) = basisstate(ComplexF64, b, indices)

"""
    sparsebasisstate([T=ComplexF64, ]b, index)

Sparse version of [`basisstate`](@ref).
"""
function sparsebasisstate(::Type{T}, b::Basis, indices) where T
    @assert length(b) == length(indices)
    x = spzeros(T, dimension(b))
    x[LinearIndices(shape(b))[indices...]] = one(T)
    Ket(b, x)
end
function sparsebasisstate(::Type{T}, b::Basis, index::Integer) where T
    data = spzeros(T, dimension(b))
    data[index] = one(T)
    Ket(b, data)
end
sparsebasisstate(b::Basis, indices) = sparsebasisstate(ComplexF64, b, indices)

SparseArrays.sparse(x::Ket) = Ket(x.basis,sparse(x.data))
SparseArrays.sparse(x::Bra) = Bra(x.basis,sparse(x.data))

# Custom broadcasting style
abstract type StateVectorStyle{B} <: Broadcast.BroadcastStyle end
struct KetStyle{B} <: StateVectorStyle{B} end
struct BraStyle{B} <: StateVectorStyle{B} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:Ket{B}}) where {B} = KetStyle{B}()
Broadcast.BroadcastStyle(::Type{<:Bra{B}}) where {B} = BraStyle{B}()
Broadcast.BroadcastStyle(::KetStyle{B1}, ::KetStyle{B2}) where {B1,B2} = throw(IncompatibleBases())
Broadcast.BroadcastStyle(::BraStyle{B1}, ::BraStyle{B2}) where {B1,B2} = throw(IncompatibleBases())

# Broadcast with scalars (of use in ODE solvers checking for tolerances, e.g. `.* reltol .+ abstol`)
Broadcast.BroadcastStyle(::T, ::Broadcast.DefaultArrayStyle{0}) where {B<:Basis, T<:KetStyle{B}} = T()
Broadcast.BroadcastStyle(::T, ::Broadcast.DefaultArrayStyle{0}) where {B<:Basis, T<:BraStyle{B}} = T()

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:KetStyle{B},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    b = find_basis(bcf)
    T = find_dType(bcf)
    data = zeros(T, dimension(b))
    @inbounds @simd for I in eachindex(bcf)
        data[I] = bcf[I]
    end
    return Ket{B}(b, data)
end
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:BraStyle{B},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    b = find_basis(bcf)
    T = find_dType(bcf)
    data = zeros(T, dimension(b))
    @inbounds @simd for I in eachindex(bcf)
        data[I] = bcf[I]
    end
    return Bra{B}(b, data)
end
for f ∈ [:find_basis,:find_dType]
    @eval ($f)(bc::Broadcast.Broadcasted) = ($f)(bc.args)
    @eval ($f)(args::Tuple) = ($f)(($f)(args[1]), Base.tail(args))
    @eval ($f)(x) = x
    @eval ($f)(::Any, rest) = ($f)(rest)
end

find_basis(x::T, rest) where {T<:Union{Ket, Bra}} = x.basis
find_dType(x::T, rest) where {T<:Union{Ket, Bra}} = eltype(x)
@inline Base.getindex(x::T, idx) where {T<:Union{Ket, Bra}} = getindex(x.data, idx)
Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(x::T, i) where {T<:Union{Ket, Bra}} = x.data[i]

# In-place broadcasting for Kets
@inline function Base.copyto!(dest::Ket{B}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:KetStyle{B},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Base.Broadcast.preprocess(dest, bc)
    dest′ = dest.data
    @inbounds @simd for I in eachindex(bc′)
        dest′[I] = bc′[I]
    end
    return dest
end
@inline Base.copyto!(dest::Ket{B1}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B1,B2,Style<:KetStyle{B2},Axes,F,Args} =
    throw(IncompatibleBases())

# In-place broadcasting for Bras
@inline function Base.copyto!(dest::Bra{B}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B,Style<:BraStyle{B},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Base.Broadcast.preprocess(dest, bc)
    dest′ = dest.data
    @inbounds @simd for I in eachindex(bc′)
        dest′[I] = bc′[I]
    end
    return dest
end
@inline Base.copyto!(dest::Bra{B1}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {B1,B2,Style<:BraStyle{B2},Axes,F,Args} =
    throw(IncompatibleBases())

@inline Base.copyto!(dest::T,src::T) where {T<:Union{Ket, Bra}} = (copyto!(dest.data,src.data); dest) # Can not use T<:QuantumInterface.StateVector, because StateVector does not imply the existence of a data property

# A few more standard interfaces: These do not necessarily make sense for a StateVector, but enable transparent use of DifferentialEquations.jl
Base.eltype(::Type{Ket{B,A}}) where {B,N,A<:AbstractVector{N}} = N # ODE init
Base.eltype(::Type{Bra{B,A}}) where {B,N,A<:AbstractVector{N}} = N
Base.any(f::Function, x::T; kwargs...) where {T<:Union{Ket, Bra}} = any(f, x.data; kwargs...) # ODE nan checks
Base.all(f::Function, x::T; kwargs...) where {T<:Union{Ket, Bra}} = all(f, x.data; kwargs...)
Base.fill!(x::T, a) where {T<:Union{Ket, Bra}} = typeof(x)(x.basis, fill!(x.data, a))
Base.similar(x::T, t) where {T<:Union{Ket, Bra}} = typeof(x)(x.basis, similar(x.data))
RecursiveArrayTools.recursivecopy!(dest::Ket{B,A},src::Ket{B,A}) where {B,A} = copyto!(dest, src) # ODE in-place equations
RecursiveArrayTools.recursivecopy!(dest::Bra{B,A},src::Bra{B,A}) where {B,A} = copyto!(dest, src)
RecursiveArrayTools.recursivecopy(x::T) where {T<:Union{Ket, Bra}} = copy(x)
RecursiveArrayTools.recursivecopy(x::AbstractArray{T}) where {T<:Union{Ket, Bra}} = copy(x)
RecursiveArrayTools.recursivefill!(x::T, a) where {T<:Union{Ket, Bra}} = fill!(x, a)
