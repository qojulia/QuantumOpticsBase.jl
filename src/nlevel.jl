import QuantumInterface: NLevelBasis

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

"""
    paulix([T=ComplexF64,] b::NLevelBasis, pow=1)

Generalized Pauli X operator for the given N level system.
Returns `X^pow`.
"""
function paulix(::Type{T}, b::NLevelBasis, pow=1) where T
    N = length(b)
    SparseOperator(b, spdiagm(pow => fill(one(T), N-pow),
                              pow-N => fill(one(T), pow)))
end
paulix(b::NLevelBasis, pow=1) = paulix(ComplexF64, b, pow)

"""
    pauliz([T=ComplexF64,] b::NLevelBasis, pow=1)

Generalized Pauli Z operator for the given N level system.
Returns `Z^pow`.
"""
function pauliz(::Type{T}, b::NLevelBasis, pow=1) where T
    N = length(b)
    ω = exp(2π*1im*pow/N)
    SparseOperator(b, spdiagm(0 => T[ω^n for n=1:N]))
end
pauliz(b::NLevelBasis, pow=1) = pauliz(ComplexF64, b, pow)

"""
    pauliy([T=ComplexF64,] b::NLevelBasis)

Pauli Y operator. Only defined for a two level system.
"""
function pauliy(::Type{T}, b::NLevelBasis) where T
    length(b) == 2 || throw(ArgumentError("pauliy only defined for two level system"))
    SparseOperator(b, spdiagm(-1 => T[-1im], 1 => T[1im]))
end
pauliy(b::NLevelBasis) = pauliy(ComplexF64,b)
