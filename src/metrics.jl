"""
    tracenorm(rho)

Trace norm of `rho`.

It is defined as

```math
T(ρ) = Tr\\{\\sqrt{ρ^† ρ}\\}.
```

Depending if `rho` is hermitian either [`tracenorm_h`](@ref) or
[`tracenorm_nh`](@ref) is called.
"""
function tracenorm(rho::DenseOpType)
    ishermitian(rho) ? tracenorm_h(rho) : tracenorm_nh(rho)
end
function tracenorm(rho::T) where T<:AbstractOperator
    throw(ArgumentError("tracenorm not implemented for $(typeof(rho)). Use dense operators instead."))
end

"""
    tracenorm_h(rho)

Trace norm of `rho`.

It uses the identity

```math
T(ρ) = Tr\\{\\sqrt{ρ^† ρ}\\} = \\sum_i |λ_i|
```

where ``λ_i`` are the eigenvalues of `rho`.
"""
function tracenorm_h(rho::DenseOpType)
    check_multiplicable(rho,rho)
    s = eigvals(Hermitian(rho.data))
    sum(abs.(s))
end
function tracenorm_h(rho::T) where T<:AbstractOperator
    throw(ArgumentError("tracenorm_h not implemented for $(typeof(rho)). Use dense operators instead."))
end


"""
    tracenorm_nh(rho)

Trace norm of `rho`.

Note that in this case `rho` doesn't have to be represented by a square
matrix (i.e. it can have different left-hand and right-hand bases).

It uses the identity

```math
    T(ρ) = Tr\\{\\sqrt{ρ^† ρ}\\} = \\sum_i σ_i
```

where ``σ_i`` are the singular values of `rho`.
"""
tracenorm_nh(rho::DenseOpType) = sum(svdvals(rho.data))
function tracenorm_nh(rho::AbstractOperator)
    throw(ArgumentError("tracenorm_nh not implemented for $(typeof(rho)). Use dense operators instead."))
end


"""
    tracedistance(rho, sigma)

Trace distance between `rho` and `sigma`.

It is defined as

```math
T(ρ,σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\}.
```

It calls [`tracenorm`](@ref) which in turn either uses [`tracenorm_h`](@ref)
or [`tracenorm_nh`](@ref) depending if ``ρ-σ`` is hermitian or not.
"""
tracedistance(rho::DenseOpType, sigma::DenseOpType) = (check_addible(rho,sigma); check_multiplicable(rho,rho); 0.5*tracenorm(rho - sigma))
function tracedistance(rho::AbstractOperator, sigma::AbstractOperator)
    throw(ArgumentError("tracedistance not implemented for $(typeof(rho)) and $(typeof(sigma)). Use dense operators instead."))
end

"""
    tracedistance_h(rho, sigma)

Trace distance between `rho` and `sigma`.

It uses the identity

```math
T(ρ,σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\} = \\frac{1}{2} \\sum_i |λ_i|
```

where ``λ_i`` are the eigenvalues of `rho` - `sigma`.
"""
tracedistance_h(rho::DenseOpType, sigma::DenseOpType) = (check_addible(rho,sigma); check_multiplicable(rho,rho); 0.5*tracenorm_h(rho - sigma))
function tracedistance_h(rho::AbstractOperator, sigma::AbstractOperator)
    throw(ArgumentError("tracedistance_h not implemented for $(typeof(rho)) and $(typeof(sigma)). Use dense operators instead."))
end

"""
    tracedistance_nh(rho, sigma)

Trace distance between `rho` and `sigma`.

Note that in this case `rho` and `sigma` don't have to be represented by square
matrices (i.e. they can have different left-hand and right-hand bases).

It uses the identity

```math
    T(ρ,σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\}
         = \\frac{1}{2} \\sum_i σ_i
```

where ``σ_i`` are the singular values of `rho` - `sigma`.
"""
tracedistance_nh(rho::DenseOpType, sigma::DenseOpType) = (check_addible(rho, sigma); 0.5*tracenorm_nh(rho - sigma))
function tracedistance_nh(rho::AbstractOperator, sigma::AbstractOperator)
    throw(ArgumentError("tracedistance_nh not implemented for $(typeof(rho)) and $(typeof(sigma)). Use dense operators instead."))
end

"""
    entropy_vn(rho)

Von Neumann entropy of a density matrix.

The Von Neumann entropy of a density operator is defined as

```math
S(ρ) = -Tr(ρ \\log(ρ)) = -\\sum_n λ_n\\log(λ_n)
```

where ``λ_n`` are the eigenvalues of the density matrix ``ρ``, ``\\log`` is the
natural logarithm and ``0\\log(0) ≡ 0``.

# Arguments
* `rho`: Density operator of which to calculate Von Neumann entropy.
* `tol=1e-15`: Tolerance for rounding errors in the computed eigenvalues.
"""
function entropy_vn(rho::DenseOpType; tol=1e-15)
    check_multiplicable(rho, rho)
    evals::Vector{ComplexF64} = eigvals(rho.data)
    entr = zero(eltype(rho))
    for d ∈ evals
        if !(abs(d) < tol)
            entr -= d*log(d)
        end
    end
    return entr
end

"""
    entropy_renyi(rho, α::Integer=2)

Renyi α-entropy of a density matrix, where r α≥0, α≂̸1.

The Renyi α-entropy of a density operator is defined as

```math
S_α(ρ) = 1/(1-α) \\log(Tr(ρ^α))
```
"""
function entropy_renyi(rho::Operator, α::Integer=2)
    α <  0 && throw(ArgumentError("α-Renyi entropy is defined for α≥0, α≂̸1"))
    α == 1 && throw(ArgumentError("α-Renyi entropy is defined for α≥0, α≂̸1"))
    check_multiplicable(rho,rho)

    return 1/(1-α) * log(tr(rho^α))
end

entropy_renyi(psi::StateVector, args...) = entropy_renyi(dm(psi), args...)

"""
    fidelity(rho, sigma)

Fidelity of two density operators.

The fidelity of two density operators ``ρ`` and ``σ`` is defined by

```math
F(ρ, σ) = Tr\\left(\\sqrt{\\sqrt{ρ}σ\\sqrt{ρ}}\\right),
```

where ``\\sqrt{ρ}=\\sum_n\\sqrt{λ_n}|ψ⟩⟨ψ|``.
"""
function fidelity(rho::DenseOpType, sigma::DenseOpType)
    check_multiplicable(rho,rho)
    check_multiplicable(sigma,sigma)
    check_multiplicable(rho,sigma)
    tr(sqrt(sqrt(rho.data)*sigma.data*sqrt(rho.data)))
end


"""
    ptranspose(rho, indices)

Partial transpose of rho with respect to subsystem specified by indices. 
                        
The `indices` argument can be a single integer or a collection of integers.
"""
function ptranspose(rho::DenseOpType, indices=1)
    length(basis_l(rho)) == length(basis_r(rho)) || throw(ArgumentError())
    length(basis_l(rho)) > 1 || throw(ArgumentError())
    # adapted from qutip.partial_transpose (https://qutip.org/docs/4.0.2/modules/qutip/partial_transpose.html)
    # works as long as QuantumOptics.jl doesn't change the implementation of `tensor`, i.e. tensor(a,b).data = kron(b.data,a.data)
    nsys = length(basis_l(rho))
    mask = ones(Int, nsys)
    mask[collect(indices)] .+= 1
    pt_dims = reshape(1:2*nsys, (nsys,2)) # indices of the operator viewed as a tensor with 2nsys legs
    pt_idx = [[pt_dims[i,mask[i]] for i = 1 : nsys]; [pt_dims[i,3-mask[i]] for i = 1 : nsys] ] # permute the legs on the subsystem of `indices`
    # reshape the operator data into a 2nsys-legged tensor and shape it back with the legs permuted
    data = reshape(permutedims(reshape(rho.data, Tuple([shape(basis_l(rho)); shape(basis_r(rho))])), pt_idx), size(rho.data))

    return DenseOperator(basis_l(rho),data)
    
end
                        


"""
    PPT(rho, index)

Peres-Horodecki criterion of partial transpose.
"""
PPT(rho::DenseOpType, index) = all(real.(eigvals(ptranspose(rho, index).data)) .>= 0.0)


"""
    negativity(rho, index)

Negativity of rho with respect to subsystem index.

The negativity of a density matrix ρ is defined as

```math
N(ρ) = \\frac{\\|ρᵀ\\|-1}{2},
```
where `ρᵀ` is the partial transpose.
"""
negativity(rho::DenseOpType, index) = 0.5*(tracenorm(ptranspose(rho, index)) - 1.0)


"""
    logarithmic_negativity(rho, index)

The logarithmic negativity of a density matrix ρ is defined as

```math
N(ρ) = \\log₂\\|ρᵀ\\|,
```
where `ρᵀ` is the partial transpose.
"""
logarithmic_negativity(rho::DenseOpType, index) = log(2, tracenorm(ptranspose(rho, index)))


"""
    avg_gate_fidelity(x, y)

The average gate fidelity between two superoperators x and y.
"""
function avg_gate_fidelity(x::T, y::T) where T <: Union{PauliTransferType, SuperOperatorType}
    check_multiplicable(x,x); check_multiplicable(y,y)
    check_samebases(basis_l(basis_l(x)), basis_r(basis_l(x)));
    dim = dimension(basis_l(basis_l(x)))
    return (tr(transpose(x.data) * y.data) + dim) / (dim^2 + dim)
end

avg_gate_fidelity(x::T, y::T) where T <: Union{ChoiStateType, ChiType} = avg_gate_fidelity(super(x), super(y))

"""
    entanglement_entropy(state, partition, [entropy_fun=entropy_vn])

Computes the entanglement entropy of `state` between the list of sites `partition`
and the rest of the system. The state must be defined in a composite basis.

If `state isa AbstractOperator` the operator-space entanglement entropy is
computed, which has the property
```julia
entanglement_entropy(dm(ket)) = 2 * entanglement_entropy(ket)
```

By default the computed entropy is the Von-Neumann entropy, but a different
function can be provided (for example to compute the entanglement-renyi entropy).
"""
entanglement_entropy(psi::Ket, partition, entropy_fun=entropy_vn) = entropy_fun(ptrace(psi, partition))

function entanglement_entropy(rho::DenseOpType, partition, args...)
    check_multiplicable(rho,rho)
    N = length(basis_l(rho))

    # build the doubled hilbert space for the vectorised dm, normalized like a Ket.
    rho_vec = normalize!(Ket(basis_l(rho)^2, vec(rho.data)))
    entanglement_entropy(rho_vec, [partition..., (partition.+N)...], args...)
end

entanglement_entropy(state::Ket, partition::Integer, args...) = entanglement_entropy(state, [partition], args...)
entanglement_entropy(state::DenseOpType, partition::Integer, args...) = entanglement_entropy(state, [partition], args...)
