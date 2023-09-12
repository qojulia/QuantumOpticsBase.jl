import ClassicalOrthogonalPolynomials: Hermite, Normalized, OrthonormalWeighted
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
function transform(::Type{S}, bp::PositionBasis, bf::FockBasis) where S
    xvec = samplepoints(bp)
    C = sqrt(spacing(bp))
    H = OrthonormalWeighted(Normalized(Hermite()))
    T = H[xvec, (bf.offset+1):(bf.N+1)]
    DenseOperator(bp, bf, convert(Matrix{S}, C*T))
end

transform(::Type{S}, bf::FockBasis, bp::PositionBasis) where S = dagger(transform(S, bp, bf))

transform(b1::Basis,b2::Basis;kwargs...) = transform(ComplexF64,b1,b2;kwargs...)
