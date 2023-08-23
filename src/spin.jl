import QuantumInterface: SpinBasis

"""
    sigmax([T=ComplexF64,] b::SpinBasis)

Pauli ``σ_x`` operator for the given Spin basis.
"""
function sigmax(::Type{T}, b::SpinBasis) where T
    N = length(b)
    diag = T[complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:N-1]
    data = spdiagm(1 => diag, -1 => diag)
    SparseOperator(b, data)
end
sigmax(b::SpinBasis) = sigmax(ComplexF64,b)

"""
    sigmay([T=ComplexF64,] b::SpinBasis)

Pauli ``σ_y`` operator for the given Spin basis.
"""
function sigmay(::Type{T}, b::SpinBasis) where T
    N = length(b)
    diag = T[1im*complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:N-1]
    data = spdiagm(-1 => diag, 1 => -diag)
    SparseOperator(b, data)
end
sigmay(b::SpinBasis) = sigmay(ComplexF64,b)

"""
    sigmaz([T=ComplexF64,] b::SpinBasis)

Pauli ``σ_z`` operator for the given Spin basis.
"""
function sigmaz(::Type{T}, b::SpinBasis) where T
    N = length(b)
    diag = T[complex(2*m) for m=b.spinnumber:-1:-b.spinnumber]
    data = spdiagm(0 => diag)
    SparseOperator(b, data)
end
sigmaz(b::SpinBasis) = sigmaz(ComplexF64,b)

"""
    sigmap([T=ComplexF64,] b::SpinBasis)

Raising operator ``σ_+`` for the given Spin basis.
"""
function sigmap(::Type{T}, b::SpinBasis) where T
    N = length(b)
    S = (b.spinnumber + 1)*b.spinnumber
    diag = T[complex(sqrt(float(S - m*(m+1)))) for m=b.spinnumber-1:-1:-b.spinnumber]
    data = spdiagm(1 => diag)
    SparseOperator(b, data)
end
sigmap(b::SpinBasis) = sigmap(ComplexF64,b)

"""
    sigmam([T=ComplexF64,] b::SpinBasis)

Lowering operator ``σ_-`` for the given Spin basis.
"""
function sigmam(::Type{T}, b::SpinBasis) where T
    N = length(b)
    S = (b.spinnumber + 1)*b.spinnumber
    diag = T[complex(sqrt(float(S - m*(m-1)))) for m=b.spinnumber:-1:-b.spinnumber+1]
    data = spdiagm(-1 => diag)
    SparseOperator(b, data)
end
sigmam(b::SpinBasis) = sigmam(ComplexF64,b)


"""
    spinup([T=ComplexF64,] b::SpinBasis)

Spin up state for the given Spin basis.
"""
spinup(::Type{T}, b::SpinBasis) where T = basisstate(T, b, 1)
spinup(b::SpinBasis) = spinup(ComplexF64, b)

"""
    spindown([T=ComplexF64], b::SpinBasis)

Spin down state for the given Spin basis.
"""
spindown(::Type{T}, b::SpinBasis) where T = basisstate(T, b, b.shape[1])
spindown(b::SpinBasis) = spindown(ComplexF64, b)


"""
    squeeze([T=ComplexF64,] b::SpinBasis, z)

Squeezing operator ``S(z)=\\exp{\\left(\\frac{z^*\\hat{J_-}^2 - z\\hat{J}_+}{2 N}\\right)}`` for the 
specified Spin-``N/2`` basis with optional data type `T`, computed as the matrix exponential. Too
large squeezing (``|z| > \\sqrt{N}``) will create an oversqueezed state.
"""
function squeeze(::Type{T}, b::SpinBasis, z::Number) where T
    N = Int(length(b)-1)
    z = T(z)/2N
    Jm = sigmam(b)/2
    Jp = sigmap(b)/2
    exp(conj(z)*dense(Jm)^2 - z*dense(Jp)^2)
end
squeeze(b::SpinBasis, z::T) where {T <: Number} = squeeze(ComplexF64, b, z)
