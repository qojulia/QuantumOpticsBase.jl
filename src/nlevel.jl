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

Generalized Pauli ``X`` operator for the given `N` level system. This is also
called the shift matrix, and generalizes the qubit pauli matrices to qudits,
while preserving the operators being unitary. A different generalization is
given by the angular momentum operators which preserves the operators being
Hermitian and is implemented for the `SpinBasis` (see [`sigmax`](@ref)).

Powers of `paulix` together with [`pauliz`](@ref) form a complete, orthornormal
(under Hilbert–Schmidt norm) operator basis.

Returns `X^pow`.
"""
function paulix(::Type{T}, b::NLevelBasis, pow=1) where T
    N = dimension(b)
    SparseOperator(b, spdiagm(pow => fill(one(T), N-pow),
                              pow-N => fill(one(T), pow)))
end
paulix(b::NLevelBasis, pow=1) = paulix(ComplexF64, b, pow)

"""
    pauliz([T=ComplexF64,] b::NLevelBasis, pow=1)

Generalized Pauli ``Z`` operator for the given `N` level system. This is also
called the clock matrix, and generalizes the qubit pauli matrices to qudits,
while preserving the operators being unitary. A different generalization is
given by the angular momentum operators which preserves the operators being
Hermitian and is implemented for the `SpinBasis` (see [`sigmaz`](@ref)).

Powers of `pauliz` together with [`paulix`](@ref) form a complete, orthornormal
(under Hilbert–Schmidt norm) operator basis.

Returns `Z^pow`.
"""
function pauliz(::Type{T}, b::NLevelBasis, pow=1) where T
    N = dimension(b)
    ω = exp(2π*1im*pow/N)
    SparseOperator(b, spdiagm(0 => T[ω^n for n=1:N]))
end
pauliz(b::NLevelBasis, pow=1) = pauliz(ComplexF64, b, pow)

"""
    pauliy([T=ComplexF64,] b::NLevelBasis)

Generalized Pauli ``Y`` operator for the given `N` level system.

Returns `Y^pow = ω^pow X^pow Z^pow` where `ω = ω = exp(2π*1im*pow/N)` and
`N = length(b)` when odd or `N = 2*length(b)` when even. This is due to the
 order of the generalized Pauli group in even versus odd dimensions.

See [`paulix`](@ref) and [`pauliz`](@ref) for more details.
"""
function pauliy(::Type{T}, b::NLevelBasis, pow=1) where T
    N = dimension(b)
    N = N%2 == 0 ? 2N : N
    ω = exp(2π*1im*pow/N)
    exp(2π*1im*pow/N)*paulix(T,b,pow)*pauliz(T,b,pow)
end

pauliy(b::NLevelBasis, pow=1) = pauliy(ComplexF64,b)
