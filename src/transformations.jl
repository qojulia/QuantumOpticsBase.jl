import FastGaussQuadrature: hermpoly_rec

"""
    transform([S=ComplexF64, ]b1::PositionBasis, b2::FockBasis; x0=1)
    transform([S=ComplexF64, ]b1::FockBasis, b2::PositionBasis; x0=1)

Transformation operator between position basis and fock basis.

The coefficients are connected via the relation
```math
ψ(x_i) = \\sum_{n=0}^N ⟨x_i|n⟩ ψ_n
```
where ``⟨x_i|n⟩`` is the value of the n-th eigenstate of a particle in
a harmonic trap potential at position ``x``, i.e.:
```math
⟨x_i|n⟩ = π^{-\\frac{1}{4}} \\frac{e^{-\\frac{1}{2}\\left(\\frac{x}{x_0}\\right)^2}}{\\sqrt{x_0}}
            \\frac{1}{\\sqrt{2^n n!}} H_n\\left(\\frac{x}{x_0}\\right)
```
"""
function transform(::Type{S}, bp::PositionBasis, bf::FockBasis; x0=1) where S
    T = Matrix{S}(undef, length(bp), length(bf))
    xvec = samplepoints(bp)
    C = pi^(-1/4)*sqrt(spacing(bp)/x0)
    for i in 1:length(bp)
        T[i,:] = C*hermpoly_rec(bf.offset:bf.N, sqrt(2)*xvec[i]/x0)
    end
    DenseOperator(bp, bf, T)
end

transform(::Type{S}, bf::FockBasis, bp::PositionBasis; x0=1) where S =
    dagger(transform(S, bp, bf; x0=x0))

transform(b1::Basis,b2::Basis;kwargs...) = transform(ComplexF64,b1,b2;kwargs...)
