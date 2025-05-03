"""
    ChargeBasis(ncut) <: Basis

Basis spanning `-ncut, ..., ncut` charge states, which are the fourier modes
(irreducible representations) of a continuous U(1) degree of freedom, truncated
at `ncut`.

The charge basis is a natural representation for circuit-QED elements such as
the "transmon", which has a hamiltonian of the form
```julia
b = ChargeBasis(ncut)
H = 4E_C * (n_g * identityoperator(b) + chargeop(b))^2 - E_J * cosφ(b)
```
with energies periodic in the charge offset `n_g`.
See e.g. https://arxiv.org/abs/2005.12667.
"""
struct ChargeBasis{T} <: Basis
    shape::Vector{T}
    dim::T
    ncut::T
    function ChargeBasis(ncut::T) where {T}
        if ncut < 0
            throw(DimensionMismatch())
        end
        dim = 2 * ncut + 1
        new{T}([dim], dim, ncut)
    end
end

Base.:(==)(b1::ChargeBasis, b2::ChargeBasis) = (b1.ncut == b2.ncut)
dimension(b::ChargeBasis) = b.dim

"""
    ShiftedChargeBasis(nmin, nmax) <: Basis

Basis spanning `nmin, ..., nmax` charge states. See [`ChargeBasis`](@ref).
"""
struct ShiftedChargeBasis{T} <: Basis
    shape::Vector{T}
    dim::T
    nmin::T
    nmax::T
    function ShiftedChargeBasis(nmin::T, nmax::T) where {T}
        if nmax <= nmin
            throw(DimensionMismatch())
        end
        dim = nmax - nmin + 1
        new{T}([dim], dim, nmin, nmax)
    end
end

Base.:(==)(b1::ShiftedChargeBasis, b2::ShiftedChargeBasis) =
    (b1.nmin == b2.nmin && b1.nmax == b2.nmax)
dimension(b::ShiftedChargeBasis) = b.dim

"""
    chargestate([T=ComplexF64,] b::ChargeBasis, n)
    chargestate([T=ComplexF64,] b::ShiftedChargeBasis, n)

Charge state ``|n⟩`` for given [`ChargeBasis`](@ref) or [`ShiftedChargeBasis`](@ref).
"""
chargestate(::Type{T}, b::ChargeBasis, n::Integer) where {T} =
    basisstate(T, b, n + b.ncut + 1)

chargestate(::Type{T}, b::ShiftedChargeBasis, n::Integer) where {T} =
    basisstate(T, b, n - b.nmin + 1)

chargestate(b, n) = chargestate(ComplexF64, b, n)

"""
    chargeop([T=ComplexF64,] b::ChargeBasis)
    chargeop([T=ComplexF64,] b::ShiftedChargeBasis)

Return diagonal charge operator ``N`` for given [`ChargeBasis`](@ref) or
[`ShiftedChargeBasis`](@ref).
"""
function chargeop(::Type{T}, b::ChargeBasis) where {T}
    data = spdiagm(T.(-b.ncut:1:b.ncut))
    return SparseOperator(b, b, data)
end

function chargeop(::Type{T}, b::ShiftedChargeBasis) where {T}
    data = spdiagm(T.(b.nmin:1:b.nmax))
    return SparseOperator(b, b, data)
end

chargeop(b) = chargeop(ComplexF64, b)

"""
    expiφ([T=ComplexF64,] b::ChargeBasis, k=1)
    expiφ([T=ComplexF64,] b::ShiftedChargeBasis, k=1)

Return operator ``\\exp(i k φ)`` for given [`ChargeBasis`](@ref) or
[`ShiftedChargeBasis`](@ref), representing the continous U(1) degree of
freedom conjugate to the charge. This is a "shift" operator that shifts
the charge by `k`.
"""
function expiφ(::Type{T}, b::ChargeBasis; k=1) where {T}
    if abs(k) > 2 * b.ncut
        data = spzeros(T, b.dim, b.dim)
    else
        v = ones(T, b.dim - abs(k))
        data = spdiagm(-k => v)
    end
    return SparseOperator(b, b, data)
end

function expiφ(::Type{T}, b::ShiftedChargeBasis; k=1) where {T}
    if abs(k) > b.dim - 1
        data = spzeros(T, b.dim, b.dim)
    else
        v = ones(T, b.dim - abs(k))
        data = spdiagm(-k => v)
    end
    return SparseOperator(b, b, data)
end

expiφ(b; kwargs...) = expiφ(ComplexF64, b; kwargs...)

"""
    cosφ([T=ComplexF64,] b::ChargeBasis; k=1)
    cosφ([T=ComplexF64,] b::ShiftedChargeBasis; k=1)

Return operator ``\\cos(k φ)`` for given charge basis. See [`expiφ`](@ref).
"""
function cosφ(::Type{T}, b::Union{ChargeBasis,ShiftedChargeBasis}; k=1) where {T}
    d = expiφ(b; k=k)
    return (d + d') / 2
end

cosφ(b; kwargs...) = cosφ(ComplexF64, b; kwargs...)

"""
    sinφ([T=ComplexF64,] b::ChargeBasis; k=1)
    sinφ([T=ComplexF64,] b::ShiftedChargeBasis; k=1)

Return operator ``\\sin(k φ)`` for given charge basis. See [`expiφ`](@ref).
"""
function sinφ(::Type{T}, b::Union{ChargeBasis,ShiftedChargeBasis}; k=1) where {T}
    d = expiφ(b; k=k)
    return (d - d') / 2im
end

sinφ(b; kwargs...) = sinφ(ComplexF64, b; kwargs...)
