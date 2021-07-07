"""
    SubspaceBasis(basisstates)

A basis describing a subspace embedded a higher dimensional Hilbert space.
"""
struct SubspaceBasis{S,B,T,H,UT} <: Basis
    shape::S
    superbasis::B
    basisstates::Vector{T}
    basisstates_hash::UT
    function SubspaceBasis{S,B,T,H,UT}(shape::S,superbasis::B,basisstates::Vector{T},basisstates_hash::UT) where {S,B,T,H,UT}
        new(shape,superbasis,basisstates,basisstates_hash)
    end
end
function SubspaceBasis(superbasis::B, basisstates::Vector{T}) where {B,T}
    for state = basisstates
        if state.basis != superbasis
            throw(ArgumentError("The basis of the basisstates has to be the superbasis."))
        end
    end
    H = hash(hash.([hash.(x.data) for x=basisstates]))
    shape = Int[length(basisstates)]
    SubspaceBasis{typeof(shape),B,T,H,typeof(H)}(shape, superbasis, basisstates, H)
end
SubspaceBasis(basisstates::Vector{T}) where T = SubspaceBasis(basisstates[1].basis, basisstates)

==(b1::SubspaceBasis, b2::SubspaceBasis) = b1.superbasis==b2.superbasis && b1.basisstates_hash==b2.basisstates_hash


proj(u::Ket, v::Ket) = dagger(v)*u/(dagger(u)*u)*u

"""
    orthonormalize(b::SubspaceBasis)

Orthonormalize the basis states of the given [`SubspaceBasis`](@ref)

A modified Gram-Schmidt process is used.
"""
function orthonormalize(b::SubspaceBasis)
    V = b.basisstates
    U = eltype(V)[]
    for (k, v)=enumerate(V)
        u = copy(v)
        for i=1:k-1
            u -= proj(U[i], u)
        end
        normalize!(u)
        push!(U, u)
    end
    return SubspaceBasis(U)
end


"""
    projector([T,] b1, b2)

Projection operator between subspaces and superspaces or between two subspaces.
"""
function projector(::Type{T}, b1::SubspaceBasis, b2::SubspaceBasis) where T
    if b1.superbasis != b2.superbasis
        throw(ArgumentError("Both subspace bases have to have the same superbasis."))
    end
    T1 = projector(T, b1, b1.superbasis)
    T2 = projector(T, b2.superbasis, b2)
    return T1*T2
end
function projector(b1::SubspaceBasis, b2::SubspaceBasis)
    if b1.superbasis != b2.superbasis
        throw(ArgumentError("Both subspace bases have to have the same superbasis."))
    end
    T1 = projector(b1, b1.superbasis)
    T2 = projector(b2.superbasis, b2)
    return T1*T2
end

function projector(::Type{T}, b1::SubspaceBasis, b2::Basis) where T
    if b1.superbasis != b2
        throw(ArgumentError("Second basis has to be the superbasis of the first one."))
    end
    data = zeros(T, length(b1), length(b2))
    for (i, state) = enumerate(b1.basisstates)
        data[i,:] = state.data
    end
    return Operator(b1, b2, data)
end
function projector(b1::SubspaceBasis, b2::Basis)
    T = promote_type(eltype.(b1.basisstates)...)
    return projector(T, b1, b2)
end

function projector(::Type{T}, b1::Basis, b2::SubspaceBasis) where T
    if b1 != b2.superbasis
        throw(ArgumentError("First basis has to be the superbasis of the second one."))
    end
    data = zeros(T, length(b1), length(b2))
    for (i, state) = enumerate(b2.basisstates)
        data[:,i] = state.data
    end
    return Operator(b1, b2, data)
end
function projector(b1::Basis, b2::SubspaceBasis)
    T = promote_type(eltype.(b2.basisstates)...)
    return projector(T, b1, b2)
end

"""
    sparseprojector([T,] b1, b2)

Sparse version of [projector](@ref).
"""
function sparseprojector(::Type{T}, b1::SubspaceBasis, b2::SubspaceBasis) where T
    if b1.superbasis != b2.superbasis
        throw(ArgumentError("Both subspace bases have to have the same superbasis."))
    end
    T1 = sparseprojector(T, b1, b1.superbasis)
    T2 = sparseprojector(T, b2.superbasis, b2)
    return T1*T2
end
function sparseprojector(b1::SubspaceBasis, b2::SubspaceBasis)
    if b1.superbasis != b2.superbasis
        throw(ArgumentError("Both subspace bases have to have the same superbasis."))
    end
    T1 = sparseprojector(b1, b1.superbasis)
    T2 = sparseprojector(b2.superbasis, b2)
    return T1*T2
end

function sparseprojector(::Type{T}, b1::SubspaceBasis, b2::Basis) where T
    if b1.superbasis != b2
        throw(ArgumentError("Second basis has to be the superbasis of the first one."))
    end
    data = spzeros(T, length(b1), length(b2))
    for (i, state) = enumerate(b1.basisstates)
        data[i,:] = state.data
    end
    return Operator(b1, b2, data)
end
function sparseprojector(b1::SubspaceBasis, b2::Basis)
    T = promote_type(eltype.(b1.basisstates)...)
    return sparseprojector(T, b1, b2)
end

function sparseprojector(::Type{T}, b1::Basis, b2::SubspaceBasis) where T
    if b1 != b2.superbasis
        throw(ArgumentError("First basis has to be the superbasis of the second one."))
    end
    data = spzeros(T, length(b1), length(b2))
    for (i, state) = enumerate(b2.basisstates)
        data[:,i] = state.data
    end
    return Operator(b1, b2, data)
end
function sparseprojector(b1::Basis, b2::SubspaceBasis)
    T = promote_type(eltype.(b2.basisstates)...)
    return sparseprojector(T, b1, b2)
end
