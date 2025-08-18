using QuantumInterface: AbstractSuperOperator

"""
    AbstractLazySuperOperator{B1,B2} <: AbstractSuperOperator{B1,B2}

Base type for lazy superoperator implementations that compute their action
on operators without explicitly storing the full superoperator matrix.
"""
abstract type AbstractLazySuperOperator{B1,B2} <: AbstractSuperOperator{B1,B2} end

"""
    LazyPrePost{B,DT} <: AbstractLazySuperOperator{Tuple{B,B},Tuple{B,B}}

Lazy superoperator that applies a pre-operator and post-operator to a quantum state/operator.
For an operator `ρ`, this computes `preop * ρ * postop'`.

# Arguments
- `preop::Operator{B,B,DT}`: Operator applied from the left
- `postop::Operator{B,B,DT}`: Operator applied from the right (conjugate transposed)

# Example
```julia
# Create Lindblad-type superoperator: L ρ L† - (1/2){L†L, ρ}
jump_op = sigmax()
dissipator = LazyPrePost(jump_op, jump_op)
evolved_state = dissipator * initial_state
```
"""
struct LazyPrePost{B,DT} <: AbstractLazySuperOperator{Tuple{B,B},Tuple{B,B}}
    preop::Operator{B,B,DT}
    postop::Operator{B,B,DT}
end

function LazyPrePost(preop::T,postop::T) where {B,DT,T<:Operator{B,B,DT}}
    LazyPrePost{B,DT}(preop,postop)
end

"""
    LazySuperSum{B,F,T} <: AbstractLazySuperOperator{Tuple{B,B},Tuple{B,B}}

Lazy sum of superoperators with corresponding factors.
Computes `sum(factor[i] * sop[i] * ρ for i in 1:length(sops))`.

# Fields
- `basis::B`: Quantum basis for the superoperator
- `factors::F`: Vector of scaling factors
- `sops::T`: Vector of superoperators to sum
"""
struct LazySuperSum{B,F,T} <: AbstractLazySuperOperator{Tuple{B,B},Tuple{B,B}}
    basis::B
    factors::F
    sops::T
end

"""
    LazySuperTensor{B,T} <: AbstractLazySuperOperator{Tuple{B,B},Tuple{B,B}}

Lazy tensor product of superoperators.
For composite quantum systems, applies each superoperator to its respective subsystem.

# Fields  
- `basis::B`: Composite quantum basis
- `sops::T`: Vector of superoperators for each subsystem
"""
struct LazySuperTensor{B,T} <: AbstractLazySuperOperator{Tuple{B,B},Tuple{B,B}}
    basis::B
    sops::T
end

# Basis functions
QuantumOpticsBase.basis(sop::LazyPrePost) = basis(sop.preop)
QuantumOpticsBase.basis(sop::LazySuperSum) = sop.basis

# Embed functions  
QuantumOpticsBase.embed(bl,br,index,op::LazyPrePost) = 
    LazyPrePost(embed(bl,br,index,op.preop), embed(bl,br,index,op.postop))

QuantumOpticsBase.embed(bl,br,index,op::LazySuperSum) = 
    LazySuperSum(bl, op.factors, [embed(bl,br,index,o) for o in op.sops])

# Application to operators
function Base.:(*)(sop::LazyPrePost, op::Operator)
    # Apply preop from left and postop from right (conjugated)
    # TODO: Optimize to avoid creating intermediate spre/spost objects
    # TODO: Implement in-place version with pre-allocated buffers
    result = op
    result = spre(sop.preop) * result  # Left multiplication
    result = spost(sop.postop) * result  # Right multiplication
    result
end

function Base.:(*)(ssop::LazySuperSum, op::Operator)
    result = zero(op)
    for (factor, sop) in zip(ssop.factors, ssop.sops)
        result += factor * (sop * op)
    end
    result
end

function Base.:(*)(ssop::LazySuperTensor, op::Operator)
    result = op
    for sop in ssop.sops
        result = sop * result
    end
    result
end

# Composition of lazy superoperators
Base.:(*)(l::LazyPrePost, r::LazyPrePost) = 
    LazyPrePost(l.preop * r.preop, r.postop * l.postop)

Base.:(+)(ops::LazyPrePost...) = 
    LazySuperSum(basis(first(ops)), fill(1, length(ops)), ops)

# Tensor product of superoperators
function QuantumInterface.tensor(sops::AbstractSuperOperator...)
    b = QuantumInterface.tensor(basis.(sops)...)
    @assert length(sops) == length(b.bases) "tensor products of superoperators over composite bases are not implemented yet"
    LazySuperTensor(b, [embed(b, b, i, s) for (i,s) in enumerate(sops)])
end