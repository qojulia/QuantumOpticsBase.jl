import Base: position
import LinearMaps
using FFTW

"""
    PositionBasis(xmin, xmax, Npoints)
    PositionBasis(b::MomentumBasis)

Basis for a particle in real space.

For simplicity periodic boundaries are assumed which means that
the rightmost point defined by `xmax` is not included in the basis
but is defined to be the same as `xmin`.

When a [`MomentumBasis`](@ref) is given as argument the exact values
of ``x_{min}`` and ``x_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dp`` and ``\\pi/dp`` with ``dp=(p_{max}-p_{min})/N``.
"""
struct PositionBasis{T,X1,X2} <: Basis
    shape::Vector{T}
    xmin::Float64
    xmax::Float64
    N::T
    function PositionBasis{X1,X2}(xmin::Real, xmax::Real, N::T) where {X1,X2,T<:Int}
        @assert isa(X1, Real) && isa(X2, Real)
        new{T,X1,X2}([N], xmin, xmax, N)
    end
end
PositionBasis(xmin::Real, xmax::Real, N::Int) = PositionBasis{xmin,xmax}(xmin,xmax,N)

"""
    MomentumBasis(pmin, pmax, Npoints)
    MomentumBasis(b::PositionBasis)

Basis for a particle in momentum space.

For simplicity periodic boundaries are assumed which means that
`pmax` is not included in the basis but is defined to be the same as `pmin`.

When a [`PositionBasis`](@ref) is given as argument the exact values
of ``p_{min}`` and ``p_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dx`` and ``\\pi/dx`` with ``dx=(x_{max}-x_{min})/N``.
"""
struct MomentumBasis{P1,P2} <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
    function MomentumBasis{P1,P2}(pmin::Real, pmax::Real, N::Int) where {P1,P2}
        @assert isa(P1, Real) && isa(P2, Real)
        new([N], pmin, pmax, N)
    end
end
MomentumBasis(pmin::Real, pmax::Real, N::Int) = MomentumBasis{pmin,pmax}(pmin, pmax, N)

PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N
==(b1::MomentumBasis, b2::MomentumBasis) = b1.pmin==b2.pmin && b1.pmax==b2.pmax && b1.N==b2.N


"""
    gaussianstate(b::PositionBasis, x0, p0, sigma)
    gaussianstate(b::MomentumBasis, x0, p0, sigma)

Create a Gaussian state around `x0` and` p0` with width `sigma`.

In real space the gaussian state is defined as

```math
\\Psi(x) = \\frac{1}{\\pi^{1/4}\\sqrt{\\sigma}}
            e^{i p_0 (x-\\frac{x_0}{2}) - \\frac{(x-x_0)^2}{2 \\sigma^2}}
```

and is connected to the momentum space definition

```math
\\Psi(p) = \\frac{\\sqrt{\\sigma}}{\\pi^{1/4}}
            e^{-i x_0 (p-\\frac{p_0}{2}) - \\frac{1}{2}(p-p_0)^2 \\sigma^2}
```

via a Fourier-transformation

```math
\\Psi(p) = \\frac{1}{\\sqrt{2\\pi}}
            \\int_{-\\infty}^{\\infty} e^{-ipx}\\Psi(x) \\mathrm{d}x
```

The state has the properties

* ``⟨p⟩ = p_0``
* ``⟨x⟩ = x_0``
* ``\\mathrm{Var}(x) = \\frac{σ^2}{2}``
* ``\\mathrm{Var}(p) = \\frac{1}{2 σ^2}``

Due to the numerically necessary discretization additional scaling
factors ``\\sqrt{Δx}`` and ``\\sqrt{Δp}`` are used so that
``\\langle x_i|Ψ\\rangle = \\sqrt{Δ x} Ψ(x_i)`` and ``\\langle p_i|Ψ\\rangle = \\sqrt{Δ p} Ψ(p_i)`` so
that the resulting Ket state is normalized.
"""
function gaussianstate(b::PositionBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dx = spacing(b)
    alpha = 1.0/(pi^(1/4)*sqrt(sigma))*sqrt(dx)
    x = b.xmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*p0*(x-x0/2) - (x-x0)^2/(2*sigma^2))
        x += dx
    end
    return psi
end

function gaussianstate(b::MomentumBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dp = spacing(b)
    alpha = sqrt(sigma)/pi^(1/4)*sqrt(dp)
    p = b.pmin
    for i=1:b.N
        psi.data[i] = alpha*exp(-1im*x0*(p-p0/2) - (p-p0)^2/2*sigma^2)
        p += dp
    end
    return psi
end


"""
    spacing(b::PositionBasis)

Difference between two adjacent points of the real space basis.
"""
spacing(b::PositionBasis) = (b.xmax - b.xmin)/b.N
"""
    spacing(b::MomentumBasis)

Momentum difference between two adjacent points of the momentum basis.
"""
spacing(b::MomentumBasis) = (b.pmax - b.pmin)/b.N

"""
    samplepoints(b::PositionBasis)

x values of the real space basis.
"""
samplepoints(b::PositionBasis) = (dx = spacing(b); Float64[b.xmin + i*dx for i=0:b.N-1])
"""
    samplepoints(b::MomentumBasis)

p values of the momentum basis.
"""
samplepoints(b::MomentumBasis) = (dp = spacing(b); Float64[b.pmin + i*dp for i=0:b.N-1])

"""
    position(b::PositionBasis)

Position operator in real space.
"""
position(b::PositionBasis) = SparseOperator(b, sparse(Diagonal(complex(samplepoints(b)))))


"""
    position(b:MomentumBasis)

Position operator in momentum space.
"""
function position(b::MomentumBasis)
    b_pos = PositionBasis(b)
    transform(b, b_pos)*dense(position(b_pos))*transform(b_pos, b)
end

"""
    momentum(b:MomentumBasis)

Momentum operator in momentum space.
"""
momentum(b::MomentumBasis) = SparseOperator(b, sparse(Diagonal(complex(samplepoints(b)))))

"""
    momentum(b::PositionBasis)

Momentum operator in real space.
"""
function momentum(b::PositionBasis)
    b_mom = MomentumBasis(b)
    transform(b, b_mom)*dense(momentum(b_mom))*transform(b_mom, b)
end

"""
    potentialoperator(b::PositionBasis, V(x))

Operator representing a potential ``V(x)`` in real space.
"""
function potentialoperator(b::PositionBasis, V::Function)
    x = samplepoints(b)
    diagonaloperator(b, V.(x))
end

"""
    potentialoperator(b::MomentumBasis, V(x))

Operator representing a potential ``V(x)`` in momentum space.
"""
function potentialoperator(b::MomentumBasis, V::Function)
    b_pos = PositionBasis(b)
    transform(b, b_pos)*dense(potentialoperator(b_pos, V))*transform(b_pos, b)
end

"""
    potentialoperator(b::CompositeBasis, V(x, y, z, ...))

Operator representing a potential ``V`` in more than one dimension.

# Arguments
* `b`: Composite basis consisting purely either of `PositionBasis` or
    `MomentumBasis`. Note, that calling this with a composite basis in
    momentum space might consume a large amount of memory.
* `V`: Function describing the potential. ATTENTION: The number of arguments
    accepted by `V` must match the spatial dimension. Furthermore, the order
    of the arguments has to match that of the order of the tensor product of
    bases (e.g. if `b=bx⊗by⊗bz`, then `V(x,y,z)`).
"""
function potentialoperator(b::CompositeBasis, V::Function)
    if isa(b.bases[1], PositionBasis)
        potentialoperator_position(b, V)
    elseif isa(b.bases[1], MomentumBasis)
        potentialoperator_momentum(b, V)
    else
        throw(IncompatibleBases())
    end
end
function potentialoperator_position(b::CompositeBasis, V::Function)
    for base=b.bases
        @assert isa(base, PositionBasis)
    end

    points = [samplepoints(b1) for b1=b.bases]
    dims = length.(points)
    n = length(b.bases)
    data = Array{ComplexF64}(undef, dims...)
    @inbounds for i=1:length(data)
        index = Tuple(CartesianIndices(data)[i])
        args = (points[j][index[j]] for j=1:n)
        data[i] = V(args...)
    end

    diagonaloperator(b, data[:])
end
function potentialoperator_momentum(b::CompositeBasis, V::Function)
    bases_pos = []
    for base=b.bases
        @assert isa(base, MomentumBasis)
        push!(bases_pos, PositionBasis(base))
    end
    b_pos = tensor(bases_pos...)
    T = tensor(transform.(b.bases, bases_pos)...)
    dense(T*dense(potentialoperator_position(b_pos, V))*T')
end

# FFT Operators
"""
    transform(b1::MomentumBasis, b2::PositionBasis)
    transform(b1::PositionBasis, b2::MomentumBasis)

Transformation operator between position basis and momentum basis.
"""
function transform(basis_l::MomentumBasis, basis_r::PositionBasis; dType=ComplexF64)
    Lx = (basis_r.xmax - basis_r.xmin)
    dp = spacing(basis_l)
    dx = spacing(basis_r)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    mul_before = exp.(-1im*basis_l.pmin*(samplepoints(basis_r) .- basis_r.xmin))
    mul_after = exp.(-1im*basis_r.xmin*samplepoints(basis_l))/sqrt(basis_r.N)
    x = Vector{dType}(undef, length(basis_r))
    fft_l! = plan_bfft!(x)
    fft_r! = plan_fft!(x)
    f!, fc! = _make_lmap_fs(basis_l.N,fft_l!,fft_r!,mul_before,mul_after)
    l = LinearMaps.LinearMap{dType}(f!,fc!,basis_l.N)
    return Operator(basis_l,basis_r,l)
end

"""
    transform(b1::CompositeBasis, b2::CompositeBasis)

Transformation operator between two composite bases. Each of the bases
has to contain bases of type PositionBasis and the other one a corresponding
MomentumBasis.
"""
function transform(basis_l::PositionBasis, basis_r::MomentumBasis; dType=ComplexF64)
    Lx = (basis_l.xmax - basis_l.xmin)
    dp = spacing(basis_r)
    dx = spacing(basis_l)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    mul_before = exp.(1im*basis_l.xmin*(samplepoints(basis_r) .- basis_r.pmin))
    mul_after = exp.(1im*basis_r.pmin*samplepoints(basis_l))/sqrt(basis_r.N)
    x = Vector{dType}(undef, length(basis_r))
    fft_l! = plan_fft!(x)
    fft_r! = plan_bfft!(x)
    f!, fc! = _make_lmap_fs(basis_l.N,fft_l!,fft_r!,mul_before,mul_after)
    l = LinearMaps.LinearMap{dType}(f!,fc!,basis_l.N)
    return Operator(basis_l,basis_r,l)
end

# Build the mutating functions for the linear maps
function _make_lmap_fs(N,fft_l!,fft_r!,mul_before,mul_after)
    function f!(result, b)
        @inbounds for i=1:N
            result[i] = mul_before[i] * b[i]
        end
        fft_r! * result
        @inbounds for i=1:N
            result[i] *= mul_after[i]
        end
        return result
    end

    function fc!(result, b)
        @inbounds for i=1:N
            result[i] = conj(mul_after[i]) * b[i]
        end
        fft_l! * result
        @inbounds for i=1:N
            result[i] = result[i] * conj(mul_before[i])
        end
        return result
    end

    return f!, fc!
end

# Skip dimension check in LinearMaps.mul! by directly calling Linearmaps._unsafe_mul!
mul!(result::Operator{B1,B3},a::Operator{B1,B2,<:LinearMaps.LinearMap},b::Operator{B2,B3},alpha,beta) where {B1<:Basis,B2<:Basis,B3<:Basis} = (LinearMaps._unsafe_mul!(result.data,a.data,b.data,alpha,beta); result)
function mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::Operator{B2,B3,<:LinearMaps.LinearMap},alpha,beta) where {B1<:Basis,B2<:Basis,B3<:Basis}
    @inbounds for i=1:size(result,1)
        # TODO: proper method for mul!(::Matrix,::Matrix,::LinearMap) ?
        y = result.data[i,:]
        x = view(a.data,i,:)
        mul!(y,transpose(b.data),x,alpha,beta)
        result.data[i,:] .= y
    end
    result
end
mul!(result::Ket{B1},a::Operator{B1,B2,<:LinearMaps.LinearMap},b::Ket{B2},alpha,beta) where {B1<:Basis,B2<:Basis} = (LinearMaps._unsafe_mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Bra{B2},a::Bra{B1},b::Operator{B1,B2,<:LinearMaps.LinearMap},alpha,beta) where {B1<:Basis,B2<:Basis} = (LinearMaps._unsafe_mul!(result.data,transpose(b.data),a.data,alpha,beta); result)

mul!(result::Operator{B1,B3},a::Operator{B1,B2,<:LinearMaps.LinearMap},b::Operator{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = (LinearMaps._unsafe_mul!(result.data,a.data,b.data); result)
function mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::Operator{B2,B3,<:LinearMaps.LinearMap}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    @inbounds for i=1:size(result,1)
        # TODO: proper method for mul!(::Matrix,::Matrix,::LinearMap) ?
        y = result.data[i,:]
        x = view(a.data,i,:)
        mul!(y,transpose(b.data),x)
        result.data[i,:] .= y
    end
    result
end
mul!(result::Ket{B1},a::Operator{B1,B2,<:LinearMaps.LinearMap},b::Ket{B2}) where {B1<:Basis,B2<:Basis} = (LinearMaps._unsafe_mul!(result.data,a.data,b.data); result)
mul!(result::Bra{B2},a::Bra{B1},b::Operator{B1,B2,<:LinearMaps.LinearMap}) where {B1<:Basis,B2<:Basis} = (LinearMaps._unsafe_mul!(result.data,transpose(b.data),a.data); result)
