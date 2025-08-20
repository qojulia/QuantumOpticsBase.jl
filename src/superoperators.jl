import Base: isapprox
import QuantumInterface: AbstractSuperOperator
import FastExpm: fastExpm
import Adapt

# TODO: this should belong in QuantumInterface.jl
abstract type OperatorBasis{BL<:Basis,BR<:Basis} end
abstract type SuperOperatorBasis{BL<:OperatorBasis,BR<:OperatorBasis} end

"""
    tensor(E::AbstractSuperOperator, F::AbstractSuperOperator, G::AbstractSuperOperator...)

Tensor product ``\\mathcal{E}⊗\\mathcal{F}⊗\\mathcal{G}⊗…`` of the given super operators.
"""
tensor(a::AbstractSuperOperator, b::AbstractSuperOperator) = arithmetic_binary_error("Tensor product", a, b)
tensor(op::AbstractSuperOperator) = op
tensor(operators::AbstractSuperOperator...) = reduce(tensor, operators)


"""
    SuperOperator <: AbstractSuperOperator

SuperOperator stored as representation, e.g. as a Matrix.
"""
mutable struct SuperOperator{B1,B2,T} <: AbstractSuperOperator{B1,B2}
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
isapprox(a::SuperOperator{B1,B2}, b::SuperOperator{B1,B2}; kwargs...) where {B1,B2} =
    (samebases(a,b) && isapprox(a.data, b.data; kwargs...))
isapprox(a::SuperOperator, b::SuperOperator; kwargs...) = false

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
    if !samebases(op.basis_l, op.basis_r)
        throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    end
    SuperOperator((op.basis_l, op.basis_l), (op.basis_r, op.basis_r), kron(identityoperator(op).data, op.data))
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
    if !samebases(op.basis_l, op.basis_r)
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

function _check_input(H::AbstractOperator{B1,B2}, J::Vector, Jdagger::Vector, rates) where {B1,B2}
    for j=J
        @assert isa(j, AbstractOperator{B1,B2})
    end
    for j=Jdagger
        @assert isa(j, AbstractOperator{B1,B2})
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

# TODO should all of PauliTransferMatrix, ChiMatrix, ChoiState, and KrausOperators subclass AbstractSuperOperator?
"""
    ChoiState <: AbstractSuperOperator

Superoperator represented as a choi state.

The convention is chosen such that the input operators live in `(basis_l[1], basis_r[1])` while
the output operators live in `(basis_r[2], basis_r[2])`.
"""
mutable struct ChoiState{B1,B2,T} <: AbstractSuperOperator{B1,B2}
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
isapprox(a::ChoiState, b::ChoiState; kwargs...) = isapprox(SuperOperator(a), SuperOperator(b); kwargs...)

# Container to hold each of the four bases for a Choi operator when converting it to
# an operator so that if any are CompositeBases tensor doesn't lossily collapse them
struct ChoiSubBasis{S,B<:Basis} <: Basis
    shape::S
    basis::B
end
ChoiSubBasis(b::Basis) = ChoiSubBasis(b.shape, b)

# TODO: decide whether to document and export this
choi_to_operator(c::ChoiState) = Operator(
    ChoiSubBasis(c.basis_l[2])⊗ChoiSubBasis(c.basis_l[1]), ChoiSubBasis(c.basis_r[2])⊗ChoiSubBasis(c.basis_r[1]), c.data)

function tensor(a::ChoiState, b::ChoiState)
    op = choi_to_operator(a) ⊗ choi_to_operator(b)
    op = permutesystems(op, [1,3,2,4])
    ChoiState((a.basis_l[1] ⊗ b.basis_l[1], a.basis_l[2] ⊗ b.basis_l[2]),
              (a.basis_r[1] ⊗ b.basis_r[1], a.basis_r[2] ⊗ b.basis_r[2]), op.data)
end
tensor(a::SuperOperator, b::SuperOperator) = SuperOperator(tensor(ChoiState(a), ChoiState(b)))

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


"""
    KrausOperators <: AbstractSuperOperator

Superoperator represented as a list of Kraus operators.

Note that KrausOperators can only represent linear maps taking density operators to other
(potentially unnormalized) density operators.
In contrast the `SuperOperator` or `ChoiState` representations can represent arbitrary linear maps
taking arbitrary operators defined on ``H_A \\to H_B`` to ``H_C \\to H_D``.
In otherwords, the Kraus representation is only defined for completely positive linear maps of the form
``(H_A \\to H_A) \\to (H_B \\to H_B)``.
Thus converting from `SuperOperator` or `ChoiState` to `KrausOperators` will throw an exception if the
map cannot be faithfully represented up to the specificed tolerance `tol`.

----------------------------
Old text:
Note unlike the SuperOperator or ChoiState types where it is possible to have
`basis_l[1] != basis_l[2]` and `basis_r[1] != basis_r[2]`
which allows representations of maps between general linear operators defined on ``H_A \\to H_B``,
a quantum channel can only act on valid density operators which live in ``H_A \\to H_A``.
Thus the Kraus representation is only defined for quantum channels which map
``(H_A \\to H_A) \\to (H_B \\to H_B)``.
"""
mutable struct KrausOperators{B1,B2,T} <: AbstractSuperOperator{B1,B2}
    basis_l::B1
    basis_r::B2
    data::Vector{T}
    function KrausOperators{BL,BR,T}(basis_l::BL, basis_r::BR, data::Vector{T}) where {BL,BR,T}
        if (any(!samebases(basis_r, M.basis_r) for M in data) ||
            any(!samebases(basis_l, M.basis_l) for M in data))
            throw(DimensionMismatch("Tried to assign data with incompatible bases"))
        end

        new(basis_l, basis_r, data)
    end
end
KrausOperators{BL,BR}(b1::BL,b2::BR,data::Vector{T}) where {BL,BR,T} = KrausOperators{BL,BR,T}(b1,b2,data)
KrausOperators(b1::BL,b2::BR,data::Vector{T}) where {BL,BR,T} = KrausOperators{BL,BR,T}(b1,b2,data)

dense(a::KrausOperators) = KrausOperators(a.basis_l, a.basis_r, [dense(op) for op in a.data])
sparse(a::KrausOperators) = KrausOperators(a.basis_l, a.basis_r, [sparse(op) for op in a.data])
dagger(a::KrausOperators) = KrausOperators(a.basis_r, a.basis_l, [dagger(op) for op in a.data])
*(a::KrausOperators{B1,B2}, b::KrausOperators{B2,B3}) where {B1,B2,B3} =
    KrausOperators(a.basis_l, b.basis_r, [A*B for A in a.data for B in b.data])
*(a::KrausOperators, b::KrausOperators) = throw(IncompatibleBases())
*(a::KrausOperators{BL,BR}, b::Operator{BR,BR}) where {BL,BR} = sum(op*b*dagger(op) for op in a.data)
==(a::KrausOperators, b::KrausOperators) = (SuperOperator(a) == SuperOperator(b))
isapprox(a::KrausOperators, b::KrausOperators; kwargs...) = isapprox(SuperOperator(a), SuperOperator(b); kwargs...)
tensor(a::KrausOperators, b::KrausOperators) =
    KrausOperators(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r,
                   [A ⊗ B for A in a.data for B in b.data])

"""
    orthogonalize(kraus::KrausOperators; tol=√eps)

Orthogonalize the set kraus operators by performing a qr decomposition on their vec'd operators.
Note that this is different than `canonicalize` which returns a kraus decomposition such
that the kraus operators are Hilbert–Schmidt orthorgonal.

If the input dimension is d and output dimension is d' then the number of kraus
operators returned is guaranteed to be no greater than dd', however it may be greater
than the Kraus rank.

`orthogonalize` should always be much faster than canonicalize as it avoids an explicit eigendecomposition
and thus also preserves sparsity if the kraus operators are sparse.
"""
function orthogonalize(kraus::KrausOperators; tol=_get_tol(kraus))
    bl, br = kraus.basis_l, kraus.basis_r
    dim = length(bl)*length(br)

    A = stack(reshape(op.data, dim) for op in kraus.data; dims=1)
    F = qr(A; tol=tol)
    # rank(F) for some reason doesn't work but should
    rank = maximum(findnz(F.R)[1]) 
    # Sanity checks that help illustrate what qr() returns:
    # @assert (F.R ≈ (sparse(F.Q') * A[F.prow,F.pcol])[1:dim,:])
    # @assert (all(iszero(F.R[rank+1:end,:])))

    ops = [Operator(bl, br, copy(reshape( # copy materializes reshaped view
        F.R[i,invperm(F.pcol)], (length(bl), length(br))))) for i=1:rank]
    return KrausOperators(bl, br, ops)
end

"""
    canonicalize(kraus::KrausOperators; tol=√eps)

Transform the quantum channel into canonical form such that the kraus operators ``{A_k}``
are Hilbert–Schmidt orthorgonal:

```math
\\Tr A_i^\\dagger A_j \\sim \\delta_{i,j}
```

If the input dimension is d and output dimension is d' then the number of kraus
operators returned is guaranteed to be no greater than dd' and will furthermore
be equal the Kraus rank of the channel up to numerical imprecision controlled by `tol`.  
"""
canonicalize(kraus::KrausOperators; tol=_get_tol(kraus)) = KrausOperators(ChoiState(kraus); tol=tol)

# TODO: document
function make_trace_preserving(kraus; tol=_get_tol(kraus))
    m = I - sum(dagger(M)*M for M in kraus.data).data
    if isa(_positive_eigen(m, tol), Number)
        throw(ArgumentError("Channel must be trace nonincreasing"))
    end
    K = Operator(kraus.basis_l, kraus.basis_r, sqrt(Matrix(m)))
    return KrausOperators(kraus.basis_l, kraus.basis_r, [kraus.data; K])
end

SuperOperator(kraus::KrausOperators) =
    SuperOperator((kraus.basis_l, kraus.basis_l), (kraus.basis_r, kraus.basis_r),
                  (sum(conj(op)⊗op for op in kraus.data)).data)

ChoiState(kraus::KrausOperators) =
    ChoiState((kraus.basis_r, kraus.basis_l), (kraus.basis_r, kraus.basis_l),
              (sum((M=op.data; reshape(M, (length(M), 1))*reshape(M, (1, length(M))))
                   for op in kraus.data)))

_choi_state_maps_density_ops(choi::ChoiState) = (samebases(choi.basis_l[1], choi.basis_r[1]) &&
                                                samebases(choi.basis_l[2], choi.basis_r[2]))

# TODO: consider using https://github.com/jlapeyre/IsApprox.jl
_is_hermitian(M, tol) = ishermitian(M) || isapprox(M, M', atol=tol)
_is_identity(M, tol) = isapprox(M, I, atol=tol)

# TODO: document
# data must be Hermitian!
function _positive_eigen(data, tol)
    # LinearAlgebra's eigen returns eigenvals sorted smallest to largest for Hermitian matrices
    vals, vecs = eigen(Hermitian(Matrix(data)))
    vals[1] < -tol && return vals[1]
    ret = [(val, vecs[:,i]) for (i, val) in enumerate(vals) if val > tol]
    return ret
end

function KrausOperators(choi::ChoiState; tol=_get_tol(choi))
    if !_choi_state_maps_density_ops(choi)
        throw(DimensionMismatch("Tried to convert Choi state of something that isn't a quantum channel mapping density operators to density operators"))
    end
    if !_is_hermitian(choi.data, tol)
        throw(ArgumentError("Tried to convert nonhermitian Choi state"))
    end
    bl, br = choi.basis_l[2], choi.basis_l[1]
    eigs = _positive_eigen(choi.data, tol)
    if isa(eigs, Number)
        throw(ArgumentError("Tried to convert a non-positive semidefinite Choi state,"*
            "failed for smallest eigval $(eigs), consider increasing tol=$(tol)"))
    end

    ops = [Operator(bl, br, sqrt(val)*reshape(vec, length(bl), length(br))) for (val, vec) in eigs]
    return KrausOperators(bl, br, ops)
end

KrausOperators(op::SuperOperator; tol=_get_tol(op)) = KrausOperators(ChoiState(op); tol=tol)

# TODO: document superoperator representation precident: everything of mixed type returns SuperOperator
*(a::ChoiState, b::SuperOperator) = SuperOperator(a)*b
*(a::SuperOperator, b::ChoiState) = a*SuperOperator(b)
*(a::KrausOperators, b::SuperOperator) = SuperOperator(a)*b
*(a::SuperOperator, b::KrausOperators) = a*SuperOperator(b)
*(a::KrausOperators, b::ChoiState) = SuperOperator(a)*SuperOperator(b)
*(a::ChoiState, b::KrausOperators) = SuperOperator(a)*SuperOperator(b)

_get_tol(kraus::KrausOperators) = sqrt(eps(real(eltype(eltype(fieldtypes(typeof(kraus))[3])))))
_get_tol(super::SuperOperator) = sqrt(eps(real(eltype(fieldtypes(typeof(super))[3]))))
_get_tol(super::ChoiState) = sqrt(eps(real(eltype(fieldtypes(typeof(super))[3]))))

# TODO: document this
is_completely_positive(choi::KrausOperators; tol=_get_tol(choi)) = true

function is_completely_positive(choi::ChoiState; tol=_get_tol(choi))
    _choi_state_maps_density_ops(choi) || return false
    _is_hermitian(choi.data, tol) || return false
    isa(_positive_eigen(Hermitian(choi.data), tol), Number) && return false
    return true
end

is_completely_positive(super::SuperOperator; tol=_get_tol(super)) =
    is_completely_positive(ChoiState(super); tol=tol)

# TODO: document this
is_trace_preserving(kraus::KrausOperators; tol=_get_tol(kraus)) =
    _is_identity(sum(dagger(M)*M for M in kraus.data).data, tol)

is_trace_preserving(choi::ChoiState; tol=_get_tol(choi)) =
    _is_identity(ptrace(choi_to_operator(choi), 1).data, tol)

is_trace_preserving(super::SuperOperator; tol=_get_tol(super)) =
    is_trace_preserving(ChoiState(super); tol=tol)

# TODO: document this
function is_trace_nonincreasing(kraus::KrausOperators; tol=_get_tol(kraus))
    m = I - sum(dagger(M)*M for M in kraus.data).data
    _is_hermitian(m, tol) || return false
    return !isa(_positive_eigen(Hermitian(m), tol), Number)
end

function is_trace_nonincreasing(choi::ChoiState; tol=_get_tol(choi))
    m = I - ptrace(choi_to_operator(choi), 1).data
    _is_hermitian(m, tol) || return false
    return !isa(_positive_eigen(Hermitian(m), tol), Number)
end

is_trace_nonincreasing(super::SuperOperator; tol=_get_tol(super)) =
    is_trace_nonincreasing(ChoiState(super); tol=tol)

# TODO: document this
is_cptp(sop; tol=_get_tol(sop)) = is_completely_positive(sop; tol=tol) && is_trace_preserving(sop; tol=tol)

# TODO: document this
is_cptni(sop; tol=_get_tol(sop)) = is_completely_positive(sop; tol=tol) && is_trace_nonincreasing(sop; tol=tol)

# GPU adaptation
Adapt.adapt_structure(to, x::SuperOperator) = SuperOperator(x.basis_l, x.basis_r, Adapt.adapt(to, x.data))

