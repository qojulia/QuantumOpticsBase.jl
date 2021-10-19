"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included
fock state (default is 0). Note that the dimension of this basis is `N+1-offset`.
"""
struct FockBasis{T} <: Basis
    shape::Vector{T}
    N::T
    offset::T
    function FockBasis(N::T,offset::T=0) where T
        if N < 0 || offset < 0 || N <= offset
            throw(DimensionMismatch())
        end
        new{T}([N-offset+1], N, offset)
    end
end


==(b1::FockBasis, b2::FockBasis) = (b1.N==b2.N && b1.offset==b2.offset)

"""
    number([T=ComplexF64,] b::FockBasis)

Number operator for the specified Fock space with optional data type `T`.
"""
function number(::Type{T}, b::FockBasis) where T
    diag = T.(b.offset:b.N)
    data = spdiagm(0 => diag)
    SparseOperator(b, data)
end

number(b::FockBasis) = number(ComplexF64, b)


"""
    destroy([T=ComplexF64,] b::FockBasis)

Annihilation operator for the specified Fock space with optional data type `T`.
"""
function destroy(::Type{C}, b::FockBasis) where C
    T = real(C)
    ns = b.offset+1.:b.N
    diag = @. C(sqrt(T(ns)))
    data = spdiagm(1 => diag)
    SparseOperator(b, data)
end

destroy(b::FockBasis) = destroy(ComplexF64, b)


"""
    create([T=ComplexF64,] b::FockBasis)

Creation operator for the specified Fock space with optional data type `T`.
"""
function create(::Type{C}, b::FockBasis) where C
    T = real(C)
    ns = b.offset+1.:b.N
    diag = @. C(sqrt(T(ns)))
    data = spdiagm(-1 => diag)
    SparseOperator(b, data)
end

create(b::FockBasis) = create(ComplexF64, b)


"""
    displace([T=ComplexF64,] b::FockBasis, alpha)

Displacement operator ``D(α)`` for the specified Fock space with optional data type `T`.
"""
function displace(::Type{T}, b::FockBasis, alpha::Number) where T
    alpha = T(alpha)
    exp(dense(alpha * create(T, b) - conj(alpha) * destroy(T, b)))
end

displace(b::FockBasis, alpha::T) where {T <: Number} = displace(ComplexF64, b, alpha)

"""
    fockstate([T=ComplexF64,] b::FockBasis, n)

Fock state ``|n⟩`` for the specified Fock space.
"""
function fockstate(::Type{T}, b::FockBasis, n::Integer) where T
    @assert b.offset <= n <= b.N
    basisstate(T, b, n+1-b.offset)
end
fockstate(b, n) = fockstate(ComplexF64, b, n)

"""
    coherentstate([T=ComplexF64,] b::FockBasis, alpha)

Coherent state ``|α⟩`` for the specified Fock space.
"""
function coherentstate(::Type{T}, b::FockBasis, alpha::Number) where T
    result = Ket(T, b)
    coherentstate!(result, b, alpha)
    return result
end
coherentstate(b, alpha) = coherentstate(ComplexF64, b, alpha)

"""
    coherentstate!(ket::Ket, b::FockBasis, alpha)

Inplace creation of coherent state ``|α⟩`` for the specified Fock space.
"""
function coherentstate!(ket::Ket, b::FockBasis, alpha::Number)
    C = eltype(ket)
    T = real(C)
    alpha = C(alpha)
    data = ket.data
    data[1] = exp(-abs2(alpha)/2)

    # Compute coefficient up to offset
    offset = b.offset
    @inbounds for n=1:offset
        data[1] *= alpha/sqrt(T(n))
    end

    # Write coefficients to state
    @inbounds for n=1:b.N-offset
        data[n+1] = data[n]*alpha/sqrt(T(n+offset))
    end

    return ket
end
