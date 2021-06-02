"""
    transform(b1::PositionBasis, b2::FockBasis; x0=1)
    transform(b1::FockBasis, b2::PositionBasis; x0=1)

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
function transform(b1::PositionBasis, b2::FockBasis; x0::X=1) where X
    T = Matrix{complex(float(X))}(undef, length(b1), length(b2))
    xvec = samplepoints(b1)
    A = hermite.A(b2.N)
    delta_x = spacing(b1)
    c = 1.0/sqrt(x0*sqrt(pi))*sqrt(delta_x)
    for i in 1:length(b1)
        u = xvec[i]/x0
        C = c*exp(-u^2/2)
        for n=b2.offset:b2.N
            idx = n-b2.offset+1
            T[i,idx] = C*horner(A[n+1], u)
        end
    end
    DenseOperator(b1, b2, T)
end

function transform(b1::FockBasis, b2::PositionBasis; x0::X=1) where X
    T = Matrix{complex(float(X))}(undef, length(b1), length(b2))
    xvec = samplepoints(b2)
    A = hermite.A(b1.N)
    delta_x = spacing(b2)
    c = 1.0/sqrt(x0*sqrt(pi))*sqrt(delta_x)
    for i in 1:length(b2)
        u = xvec[i]/x0
        C = c*exp(-u^2/2)
        for n=b1.offset:b1.N
            idx = n-b1.offset+1
            T[idx,i] = C*horner(A[n+1], u)
        end
    end
    DenseOperator(b1, b2, T)
end
