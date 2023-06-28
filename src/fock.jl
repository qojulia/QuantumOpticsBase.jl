import QuantumInterface: FockBasis

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

Displacement operator ``D(α)`` for the specified Fock space with optional data
type `T`, computed as the matrix exponential of finite-dimensional (truncated)
creation and annihilation operators.
"""
function displace(::Type{T}, b::FockBasis, alpha::Number) where T
    alpha = T(alpha)
    exp(dense(alpha * create(T, b) - conj(alpha) * destroy(T, b)))
end

displace(b::FockBasis, alpha::T) where {T <: Number} = displace(ComplexF64, b, alpha)

"""
    squeeze([T=ComplexF64,] b::FockBasis, z)

Squeezing operator ``S(z)`` for the specified Fock space with optional data
type `T`, computed as the matrix exponential of finite-dimensional (truncated)
creation and annihilation operators.
"""
function squeeze(::Type{T}, b::FockBasis, z::Number) where T
    z = T(z)/2
    asq = destroy(T, b)^2
    exp(dense(conj(z) * asq - z * dagger(asq)))
end

squeeze(b::FockBasis, z::T) where {T <: Number} = squeeze(ComplexF64, b, z)

# associated Laguerre polynomial, borrowed from IonSim.jl
function _alaguerre(x::Real, n::Int, k::Int)
    L = 1.0, -x + k + 1
    if n < 2
        return L[n + 1]
    end
    for i in 2:n
        L = L[2], ((k + 2i - 1 - x) * L[2] - (k + i - 1) * L[1]) / i
    end
    return L[2]
end

"""
    displace_analytical(alpha::Number, n::Integer, m::Integer)

Get a specific matrix element of the (analytical) displacement operator in the
Fock basis: `Dmn = ⟨n|D̂(α)|m⟩`. The precision used for computation is based
on the type of `alpha`. If `alpha` is a Float64, ComplexF64, or Int, the computation
will be carried out at double precision.
"""
function displace_analytical(alpha::Number, n::Integer, m::Integer)
    # Borrowed from IonSim.jl.
    if n < m
        return (-1)^isodd(abs(n - m)) * conj(displace_analytical(alpha, m, n))
    end
    # compute factorial ratio directly, in float representation, to avoid integer overflow
    s = 1.0 * one(real(alpha))
    for i in (m + 1):n
        s *= i
    end
    ret = sqrt(1 / s) * alpha^(n - m) * exp(-abs2(alpha) / 2.0) * _alaguerre(abs2(alpha), m, n - m)
    if isnan(ret)
        # Handles factorial -> Inf in case of large n-m, and also large abs2(alpha) making _alaguerre() return NaN.
        return 1.0 * (n == m)
    end
    return ret
end

"""
    displace_analytical!(op, alpha::Number)

Overwrite, in place, the matrix elements of the FockBasis operator `op`, so that
it is equal to `displace_analytical(eltype(op), basis(op), alpha)`
"""
function displace_analytical!(op::DataOperator{B,B}, alpha::Number) where {B<:FockBasis}
    b = basis(op)
    ofs = b.offset
    @inbounds for n in 1:size(op.data, 2), m in 1:size(op.data, 1)
          op.data[m, n] = displace_analytical(alpha, m-1 + ofs, n-1 + ofs)
    end
    op
end

"""
    displace_analytical(b::FockBasis, alpha::Number)
    displace_analytical(::Type{T}, b::FockBasis, alpha::Number)

Get the "analytical" displacement operator, whose matrix elements match (up to
numerical imprecision) those of the exact infinite-dimensional displacement
operator. This is different to the result of `displace(..., alpha)`, which
computes the matrix exponential `exp(alpha * a' - conj(alpha) * a)` using
finite-dimensional (truncated) creation and annihilation operators `a'` and `a`.
"""
function displace_analytical(::Type{T}, b::FockBasis, alpha::Number) where T
    displace_analytical!(DenseOperator(T, b), alpha)
end

displace_analytical(b::FockBasis, alpha::Number) = displace_analytical(ComplexF64, b::FockBasis, alpha::Number)


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
