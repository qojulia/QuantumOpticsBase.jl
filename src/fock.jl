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
    number(b::FockBasis)

Number operator for the specified Fock space.
"""
function number(b::FockBasis)
    diag = complex.(b.offset:b.N)
    data = spdiagm(0 => diag)
    SparseOperator(b, data)
end

"""
    destroy(b::FockBasis)

Annihilation operator for the specified Fock space.
"""
function destroy(b::FockBasis)
    diag = complex.(sqrt.(b.offset+1.:b.N))
    data = spdiagm(1 => diag)
    SparseOperator(b, data)
end

"""
    create(b::FockBasis)

Creation operator for the specified Fock space.
"""
function create(b::FockBasis)
    diag = complex.(sqrt.(b.offset+1.:b.N))
    data = spdiagm(-1 => diag)
    SparseOperator(b, data)
end

"""
    displace(b::FockBasis, alpha)

Displacement operator ``D(α)`` for the specified Fock space.
"""
displace(b::FockBasis, alpha) = exp(dense(alpha*create(b) - conj(alpha)*destroy(b)))

"""
    fockstate(b::FockBasis, n)

Fock state ``|n⟩`` for the specified Fock space.
"""
function fockstate(b::FockBasis, n)
    @assert b.offset <= n <= b.N
    basisstate(b, n+1-b.offset)
end

"""
    coherentstate(b::FockBasis, alpha)

Coherent state ``|α⟩`` for the specified Fock space.
"""
function coherentstate(b::FockBasis, alpha)
    result = Ket(b, Vector{ComplexF64}(undef, length(b)))
    coherentstate!(result, b, alpha)
    return result
end

"""
    coherentstate!(ket::Ket, b::FockBasis, alpha)

Inplace creation of coherent state ``|α⟩`` for the specified Fock space.
"""
function coherentstate!(ket::Ket, b::FockBasis, alpha)
    data = ket.data
    data[1] = exp(-abs2(alpha)/2)

    # Compute coefficient up to offset
    offset = b.offset
    @inbounds for n=1:offset
        data[1] *= alpha/sqrt(n)
    end

    # Write coefficients to state
    @inbounds for n=1:b.N-offset
        data[n+1] = data[n]*alpha/sqrt(n+offset)
    end

    return ket
end
