"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
struct NLevelBasis{T} <: Basis
    shape::Vector{T}
    N::T
    function NLevelBasis(N::T) where T<:Integer
        if N < 1
            throw(DimensionMismatch())
        end
        new{T}([N], N)
    end
end

==(b1::NLevelBasis, b2::NLevelBasis) = b1.N == b2.N


"""
    transition([T=ComplexF64,] b::NLevelBasis, to::Integer, from::Integer)

Transition operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|``.
"""
function transition(::Type{T}, b::NLevelBasis, to::Integer, from::Integer) where T
    if to < 1 || b.N < to
        throw(BoundsError("'to' index has to be between 1 and b.N"))
    end
    if from < 1 || b.N < from
        throw(BoundsError("'from' index has to be between 1 and b.N"))
    end
    op = SparseOperator(T, b)
    op.data[to, from] = 1.
    op
end
transition(b::NLevelBasis,to::Integer,from::Integer) = transition(ComplexF64,b,to,from)


"""
    nlevelstate([T=ComplexF64,] b::NLevelBasis, n::Integer)

State where the system is completely in the n-th level.
"""
function nlevelstate(::Type{T}, b::NLevelBasis, n::Integer) where T
    if n < 1 || b.N < n
        throw(BoundsError("n has to be between 1 and b.N"))
    end
    basisstate(T, b, n)
end
nlevelstate(b::NLevelBasis, n::Integer) = nlevelstate(ComplexF64, b, n)
