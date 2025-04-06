import QuantumInterface: AbstractSuperOperator
import FastExpm: fastExpm

"""
    SuperOperator <: AbstractSuperOperator

SuperOperator stored as representation, e.g. as a Matrix.
"""
mutable struct SuperOperator{B1,B2,T} <: AbstractSuperOperator
    basis_l::B1
    basis_r::B2
    data::T
    function SuperOperator{BL,BR,T}(basis_l::BL, basis_r::BR, data::T) where {BL,BR,T}
        if (length(basis_l) != 2 || length(basis_r) != 2 ||
            length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
            length(basis_r[1])*length(basis_r[2]) != size(data, 2))
            throw(DimensionMismatch("Tried to assign data of size $(size(data)) to Hilbert spaces of sizes $(length.(basis_l)), $(length.(basis_r))"))
        end
        new(basis_l, basis_r, data)
    end
end
SuperOperator{BL,BR}(b1::BL,b2::BR,data::T) where {BL,BR,T} = SuperOperator{BL,BR,T}(b1,b2,data)
SuperOperator(b1::BL,b2::BR,data::T) where {BL,BR,T} = SuperOperator{BL,BR,T}(b1,b2,data)
SuperOperator(b,data) = SuperOperator(b,b,data)

const DenseSuperOpType{BL,BR} = SuperOperator{BL,BR,<:Matrix}
const SparseSuperOpType{BL,BR} = SuperOperator{BL,BR,<:SparseMatrixCSC}

"""
    DenseSuperOperator(b1[, b2, data])
    DenseSuperOperator([T=ComplexF64,], b1[, b2])

SuperOperator stored as dense matrix.
"""
DenseSuperOperator(basis_l,basis_r,data) = SuperOperator(basis_l, basis_r, Matrix(data))
function DenseSuperOperator(::Type{T}, basis_l, basis_r) where T
    Nl = length(basis_l[1])*length(basis_l[2])
    Nr = length(basis_r[1])*length(basis_r[2])
    data = zeros(T, Nl, Nr)
    DenseSuperOperator(basis_l, basis_r, data)
end
DenseSuperOperator(basis_l, basis_r) = DenseSuperOperator(ComplexF64, basis_l, basis_r)
DenseSuperOperator(::Type{T}, b) where T = DenseSuperOperator(T, b, b)
DenseSuperOperator(b) = DenseSuperOperator(b,b)


"""
    SparseSuperOperator(b1[, b2, data])
    SparseSuperOperator([T=ComplexF64,], b1[, b2])

SuperOperator stored as sparse matrix.
"""
SparseSuperOperator(basis_l, basis_r, data) = SuperOperator(basis_l, basis_r, sparse(data))

function SparseSuperOperator(::Type{T}, basis_l, basis_r) where T
    Nl = length(basis_l[1])*length(basis_l[2])
    Nr = length(basis_r[1])*length(basis_r[2])
    data = spzeros(T, Nl, Nr)
    SparseSuperOperator(basis_l, basis_r, data)
end
SparseSuperOperator(basis_l, basis_r) = SparseSuperOperator(ComplexF64, basis_l, basis_r)
SparseSuperOperator(::Type{T}, b) where T = SparseSuperOperator(T, b, b)
SparseSuperOperator(b) = DenseSuperOperator(b,b)

Base.copy(a::T) where {T<:SuperOperator} = T(a.basis_l, a.basis_r, copy(a.data))

dense(a::SuperOperator) = DenseSuperOperator(a.basis_l, a.basis_r, a.data)
sparse(a::SuperOperator) = SuperOperator(a.basis_l, a.basis_r, sparse(a.data))

==(a::SuperOperator{B1,B2}, b::SuperOperator{B1,B2}) where {B1,B2} = (samebases(a,b) && a.data == b.data)
==(a::SuperOperator, b::SuperOperator) = false

Base.length(a::SuperOperator) = length(a.basis_l[1])*length(a.basis_l[2])*length(a.basis_r[1])*length(a.basis_r[2])
samebases(a::SuperOperator, b::SuperOperator) = samebases(a.basis_l[1], b.basis_l[1]) && samebases(a.basis_l[2], b.basis_l[2]) &&
                                                      samebases(a.basis_r[1], b.basis_r[1]) && samebases(a.basis_r[2], b.basis_r[2])
multiplicable(a::SuperOperator, b::SuperOperator) = multiplicable(a.basis_r[1], b.basis_l[1]) && multiplicable(a.basis_r[2], b.basis_l[2])
multiplicable(a::SuperOperator, b::AbstractOperator) = multiplicable(a.basis_r[1], b.basis_l) && multiplicable(a.basis_r[2], b.basis_r)


# Arithmetic operations
function *(a::SuperOperator{B1,B2}, b::Operator{BL,BR}) where {BL,BR,B1,B2<:Tuple{BL,BR}}
    data = a.data*reshape(b.data, length(b.data))
    return Operator(a.basis_l[1], a.basis_l[2], reshape(data, length(a.basis_l[1]), length(a.basis_l[2])))
end

function *(a::SuperOperator{B1,B2}, b::SuperOperator{B2,B3}) where {B1,B2,B3}
    return SuperOperator{B1,B3}(a.basis_l, b.basis_r, a.data*b.data)
end

*(a::SuperOperator, b::Number) = SuperOperator(a.basis_l, a.basis_r, a.data*b)
*(a::Number, b::SuperOperator) = b*a

/(a::SuperOperator, b::Number) = SuperOperator(a.basis_l, a.basis_r, a.data ./ b)

+(a::SuperOperator{B1,B2}, b::SuperOperator{B1,B2}) where {B1,B2} = SuperOperator{B1,B2}(a.basis_l, a.basis_r, a.data+b.data)
+(a::SuperOperator, b::SuperOperator) = throw(IncompatibleBases())

-(a::SuperOperator{B1,B2}, b::SuperOperator{B1,B2}) where {B1,B2} = SuperOperator{B1,B2}(a.basis_l, a.basis_r, a.data-b.data)
-(a::SuperOperator) = SuperOperator(a.basis_l, a.basis_r, -a.data)
-(a::SuperOperator, b::SuperOperator) = throw(IncompatibleBases())

identitysuperoperator(b::Basis) =
    SuperOperator((b,b), (b,b), Eye{ComplexF64}(length(b)^2))

identitysuperoperator(op::DenseSuperOpType) = 
    SuperOperator(op.basis_l, op.basis_r, Matrix(one(eltype(op.data))I, size(op.data)))

identitysuperoperator(op::SparseSuperOpType) = 
    SuperOperator(op.basis_l, op.basis_r, sparse(one(eltype(op.data))I, size(op.data)))

dagger(x::DenseSuperOpType) = SuperOperator(x.basis_r, x.basis_l, copy(adjoint(x.data)))
dagger(x::SparseSuperOpType) = SuperOperator(x.basis_r, x.basis_l, sparse(adjoint(x.data)))


"""
    spre(op)

Create a super-operator equivalent for right side operator multiplication.

For operators ``A``, ``B`` the relation

```math
    \\mathrm{spre}(A) B = A B
```

holds. `op` can be a dense or a sparse operator.
"""
function spre(op::AbstractOperator)
    if !multiplicable(op, op)
        throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    end
    SuperOperator((op.basis_l, op.basis_l), (op.basis_r, op.basis_r), tensor(op, identityoperator(op)).data)
end

"""
    spost(op)

Create a super-operator equivalent for left side operator multiplication.

For operators ``A``, ``B`` the relation

```math
    \\mathrm{spost}(A) B = B A
```

holds. `op` can be a dense or a sparse operator.
"""
function spost(op::AbstractOperator)
    if !multiplicable(op, op)
        throw(ArgumentError("It's not clear what spost of a non-square operator should be. See issue #113"))
    end
    SuperOperator((op.basis_r, op.basis_r), (op.basis_l, op.basis_l), kron(permutedims(op.data), identityoperator(op).data))
end

"""
    sprepost(op)

Create a super-operator equivalent for left and right side operator multiplication.

For operators ``A``, ``B``, ``C`` the relation

```math
    \\mathrm{sprepost}(A, B) C = A C B
```

holds. `A` ond `B` can be dense or a sparse operators.
"""
sprepost(A::AbstractOperator, B::AbstractOperator) = SuperOperator((A.basis_l, B.basis_r), (A.basis_r, B.basis_l), kron(permutedims(B.data), A.data))

function _check_input(H::AbstractOperator, J::Vector, Jdagger::Vector, rates)
    for j=J
        @assert isa(j, AbstractOperator)
    end
    for j=Jdagger
        @assert isa(j, AbstractOperator)
    end
    @assert length(J)==length(Jdagger)
    if isa(rates, Matrix{<:Number})
        @assert size(rates, 1) == size(rates, 2) == length(J)
    elseif isa(rates, Vector{<:Number})
        @assert length(rates) == length(J)
    end
end


"""
    liouvillian(H, J; rates, Jdagger)

Create a super-operator equivalent to the master equation so that ``\\dot ρ = S ρ``.

The super-operator ``S`` is defined by

```math
S ρ = -\\frac{i}{ħ} [H, ρ] + \\sum_i J_i ρ J_i^† - \\frac{1}{2} J_i^† J_i ρ - \\frac{1}{2} ρ J_i^† J_i
```

# Arguments
* `H`: Hamiltonian.
* `J`: Vector containing the jump operators.
* `rates`: Vector or matrix specifying the coefficients for the jump operators.
* `Jdagger`: Vector containing the hermitian conjugates of the jump operators. If they
             are not given they are calculated automatically.
"""
function liouvillian(H, J; rates=ones(length(J)), Jdagger=dagger.(J))
    _check_input(H, J, Jdagger, rates)
    L = spre(-1im*H) + spost(1im*H)
    if isa(rates, AbstractMatrix)
        for i=1:length(J), j=1:length(J)
            jdagger_j = rates[i,j]/2*Jdagger[j]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(rates[i,j]*J[i]) * spost(Jdagger[j])
        end
    elseif isa(rates, AbstractVector)
        for i=1:length(J)
            jdagger_j = rates[i]/2*Jdagger[i]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(rates[i]*J[i]) * spost(Jdagger[i])
        end
    end
    return L
end

"""
    exp(op::DenseSuperOperator)

Superoperator exponential which can, for example, be used to calculate time evolutions.
Uses LinearAlgebra's `Base.exp`.

If you only need the result of the exponential acting on an operator,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
Base.exp(op::DenseSuperOpType) = DenseSuperOperator(op.basis_l, op.basis_r, exp(op.data))

"""
    exp(op::SparseSuperOperator; opts...)

Superoperator exponential which can, for example, be used to calculate time evolutions.
Uses [`FastExpm.jl.jl`](https://github.com/fmentink/FastExpm.jl) which will return a sparse
or dense operator depending on which is more efficient.
All optional arguments are passed to `fastExpm` and can be used to specify tolerances.

If you only need the result of the exponential acting on an operator,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
function Base.exp(op::SparseSuperOpType; opts...)
    if iszero(op)
        return identitysuperoperator(op)
    else
        return SuperOperator(op.basis_l, op.basis_r, fastExpm(op.data; opts...))
    end
end

# Array-like functions
Base.zero(A::SuperOperator) = SuperOperator(A.basis_l, A.basis_r, zero(A.data))
Base.size(A::SuperOperator) = size(A.data)
@inline Base.axes(A::SuperOperator) = axes(A.data)
Base.ndims(A::SuperOperator) = 2
Base.ndims(::Type{<:SuperOperator}) = 2

# Broadcasting
Base.broadcastable(A::SuperOperator) = A

# Custom broadcasting styles
struct SuperOperatorStyle{BL,BR} <: Broadcast.BroadcastStyle end
# struct DenseSuperOperatorStyle{BL,BR} <: SuperOperatorStyle{BL,BR} end
# struct SparseSuperOperatorStyle{BL,BR} <: SuperOperatorStyle{BL,BR} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:SuperOperator{BL,BR}}) where {BL,BR} = SuperOperatorStyle{BL,BR}()
# Broadcast.BroadcastStyle(::Type{<:SparseSuperOperator{BL,BR}}) where {BL,BR} = SparseSuperOperatorStyle{BL,BR}()
# Broadcast.BroadcastStyle(::DenseSuperOperatorStyle{B1,B2}, ::SparseSuperOperatorStyle{B1,B2}) where {B1,B2} = DenseSuperOperatorStyle{B1,B2}()
# Broadcast.BroadcastStyle(::DenseSuperOperatorStyle{B1,B2}, ::DenseSuperOperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
# Broadcast.BroadcastStyle(::SparseSuperOperatorStyle{B1,B2}, ::SparseSuperOperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
# Broadcast.BroadcastStyle(::DenseSuperOperatorStyle{B1,B2}, ::SparseSuperOperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:SuperOperatorStyle{BL,BR},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    return SuperOperator{BL,BR}(bl, br, copy(bc_))
end
# @inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:SparseSuperOperatorStyle{BL,BR},Axes,F,Args<:Tuple}
#     bcf = Broadcast.flatten(bc)
#     bl,br = find_basis(bcf.args)
#     bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
#     return SuperOperator{BL,BR}(bl, br, copy(bc_))
# end
find_basis(a::SuperOperator, rest) = (a.basis_l, a.basis_r)

const BasicMathFunc = Union{typeof(+),typeof(-),typeof(*),typeof(/)}
function Broadcasted_restrict_f(f::BasicMathFunc, args::Tuple{Vararg{<:SuperOperator}}, axes)
    args_ = Tuple(a.data for a=args)
    return Broadcast.Broadcasted(f, args_, axes)
end

# In-place broadcasting
@inline function Base.copyto!(dest::SuperOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:SuperOperatorStyle{BL,BR},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && isa(bc.args, Tuple{<:SuperOperator{BL,BR}}) # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    # Get the underlying data fields of operators and broadcast them as arrays
    bcf = Broadcast.flatten(bc)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    copyto!(dest.data, bc_)
    return dest
end
@inline Base.copyto!(A::SuperOperator{BL,BR},B::SuperOperator{BL,BR}) where {BL,BR} = (copyto!(A.data,B.data); A)
@inline function Base.copyto!(dest::SuperOperator{B1,B2}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {
        B1,B2,B3,
        B4,Style<:SuperOperatorStyle{B3,B4},Axes,F,Args
        }
    throw(IncompatibleBases())
end

"""
    ChoiState <: AbstractSuperOperator

Superoperator represented as a choi state.
"""
mutable struct ChoiState{B1,B2,T} <: AbstractSuperOperator
    basis_l::B1
    basis_r::B2
    data::T
    function ChoiState{BL,BR,T}(basis_l::BL, basis_r::BR, data::T) where {BL,BR,T}
        if (length(basis_l) != 2 || length(basis_r) != 2 ||
            length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
            length(basis_r[1])*length(basis_r[2]) != size(data, 2))
            throw(DimensionMismatch("Tried to assign data of size $(size(data)) to Hilbert spaces of sizes $(length.(basis_l)), $(length.(basis_r))"))
        end
        new(basis_l, basis_r, data)
    end
end
ChoiState(b1::BL, b2::BR, data::T) where {BL,BR,T} = ChoiState{BL,BR,T}(b1, b2, data)

dense(a::ChoiState) = ChoiState(a.basis_l, a.basis_r, Matrix(a.data))
sparse(a::ChoiState) = ChoiState(a.basis_l, a.basis_r, sparse(a.data))
dagger(a::ChoiState) = ChoiState(dagger(SuperOperator(a)))
*(a::ChoiState, b::ChoiState) = ChoiState(SuperOperator(a)*SuperOperator(b))
*(a::ChoiState, b::Operator) = SuperOperator(a)*b
==(a::ChoiState, b::ChoiState) = (SuperOperator(a) == SuperOperator(b))

# reshape swaps within systems due to colum major ordering
# https://docs.qojulia.org/quantumobjects/operators/#tensor_order
function _super_choi((l1, l2), (r1, r2), data)
    data = Base.ReshapedArray(data, map(length, (l2, l1, r2, r1)), ())
    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    data = PermutedDimsArray(data, (1, 3, 2, 4))
    data = Base.ReshapedArray(data, map(length, (l1⊗l2, r1⊗r2)), ())
    return (l1, l2), (r1, r2), copy(data)
end

ChoiState(op::SuperOperator) = ChoiState(_super_choi(op.basis_l, op.basis_r, op.data)...)
SuperOperator(op::ChoiState) = SuperOperator(_super_choi(op.basis_l, op.basis_r, op.data)...)

*(a::ChoiState, b::SuperOperator) = SuperOperator(a)*b
*(a::SuperOperator, b::ChoiState) = a*SuperOperator(b)

