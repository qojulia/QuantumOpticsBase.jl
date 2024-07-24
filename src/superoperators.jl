import QuantumInterface: AbstractSuperOperator
import FastExpm: fastExpm

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

const BasicMathFunc = Union{typeof(+),typeof(-),typeof(*)}
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
    KrausOperators(B1, B2, data)

Superoperator represented as a list of Kraus operators.
Note unlike the SuperOperator or ChoiState types where
its possible to have basis_l[1] != basis_l[2] and basis_r[1] != basis_r[2]
which allows representations of maps between general linear operators defined on H_A \to H_B,
a quantum channel can only act on valid density operators which live in H_A \to H_A.
Thus the Kraus representation is only defined for quantum channels which map
(H_A \to H_A) \to (H_B \to H_B).
    """
mutable struct KrausOperators{B1,B2,T} <: AbstractSuperOperator{B1,B2}
    basis_l::B1
    basis_r::B2
    data::T
    function KrausOperators{BL,BR,T}(basis_l::BL, basis_r::BR, data::T) where {BL,BR,T}
        if (any(!samebases(basis_r, M.basis_r) for M in data) ||
            any(!samebases(basis_l, M.basis_l) for M in data))
            throw(DimensionMismatch("Tried to assign data with incompatible bases"))
        end

        new(basis_l, basis_r, data)
    end
end
KrausOperators{BL,BR}(b1::BL,b2::BR,data::T) where {BL,BR,T} = KrausOperators{BL,BR,T}(b1,b2,data)
KrausOperators(b1::BL,b2::BR,data::T) where {BL,BR,T} = KrausOperators{BL,BR,T}(b1,b2,data)
KrausOperators(b,data) = KrausOperators(b,b,data)

function is_trace_preserving(kraus::KrausOperators; tol=1e-9)
    m = I(length(kraus.basis_r)) - sum(dagger(M)*M for M in kraus.data).data
    m[abs.(m) .< tol] .= 0
    return iszero(m)
end

function is_valid_channel(kraus::KrausOperators; tol=1e-9)
    m = I(length(kraus.basis_r)) - sum(dagger(M)*M for M in kraus.data).data
    eigs = eigvals(Matrix(m))
    eigs[@. abs(eigs) < tol || eigs > 0] .= 0
    return iszero(eigs)
end

"""
    ChoiState(B1, B2, data)

Superoperator represented as a choi state stored as a sparse or dense matrix.
"""
mutable struct ChoiState{B1,B2,T} <: AbstractSuperOperator{B1,B2}
    basis_l::B1
    basis_r::B2
    data::T
    function ChoiState{BL,BR,T}(basis_l::BL, basis_r::BR, data::T; tol=1e-9) where {BL,BR,T}
        if (length(basis_l) != 2 || length(basis_r) != 2 ||
            length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
            length(basis_r[1])*length(basis_r[2]) != size(data, 2))
            throw(DimensionMismatch("Tried to assign data of size $(size(data)) to Hilbert spaces of sizes $(length.(basis_l)), $(length.(basis_r))"))
        end
        if any(abs.(data - data') .> tol)
            @warn "Trying to construct ChoiState from non-hermitian data"
        end
        new(basis_l, basis_r, Hermitian(data))
    end
end
ChoiState{BL,BR}(b1::BL,b2::BR,data::T; tol=1e-9) where {BL,BR,T} = ChoiState{BL,BR,T}(b1,b2,data;tol=tol)
ChoiState(b1::BL,b2::BR,data::T; tol=1e-9) where {BL,BR,T} = ChoiState{BL,BR,T}(b1,b2,data; tol=tol)
ChoiState(b,data; tol=tol) = ChoiState(b,b,data; tol=tol)

# TODO: document why we have super_to_choi return non-trace one density matrices.
# https://forest-benchmarking.readthedocs.io/en/latest/superoperator_representations.html
# Note the similarity to permutesystems in operators_dense.jl

# reshape swaps within systems due to colum major ordering
# https://docs.qojulia.org/quantumobjects/operators/#tensor_order
function _super_choi((l1, l2), (r1, r2), data::Matrix)
    data = reshape(data, map(length, (l2, l1, r2, r1)))
    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    data = permutedims(data, (1, 3, 2, 4))
    data = reshape(data, map(length, (l1⊗l2, r1⊗r2)))
    return (l1, l2), (r1, r2), data
end

function _super_choi((r2, l2), (r1, l1), data::SparseMatrixCSC)
    data = _permutedims(data, map(length, (l2, r2, l1, r1)), (1, 3, 2, 4))
    data = reshape(data, map(length, (l1⊗l2, r1⊗r2)))
    # sparse(data) is necessary since reshape of a sparse array returns a
    # ReshapedSparseArray which is not a subtype of AbstractArray and so
    # _permutedims fails to acces the ".m" field
    # https://github.com/qojulia/QuantumOpticsBase.jl/pull/83
    # https://github.com/JuliaSparse/SparseArrays.jl/issues/24
    # permutedims in SparseArrays.jl only implements perm (2,1) and so
    # _permutedims should be upstreamed
    # https://github.com/JuliaLang/julia/issues/26534
    return (l1, l2), (r1, r2), sparse(data)
end

ChoiState(op::SuperOperator; tol=1e-9) = ChoiState(_super_choi(op.basis_l, op.basis_r, op.data)...; tol=tol)
SuperOperator(kraus::KrausOperators) =
    SuperOperator((kraus.basis_l, kraus.basis_l), (kraus.basis_r, kraus.basis_r),
                  (sum(conj(op)⊗op for op in kraus.data)).data)

SuperOperator(op::ChoiState) = SuperOperator(_super_choi(op.basis_l, op.basis_r, op.data)...)
ChoiState(kraus::KrausOperators) = ChoiState(SuperOperator(kraus))

function KrausOperators(choi::ChoiState; tol=1e-9)
    if (!samebases(choi.basis_l[1], choi.basis_l[2]) ||
        !samebases(choi.basis_r[1], choi.basis_r[2]))
        throw(DimensionMismatch("Tried to convert choi state of something that isn't a quantum channel mapping density operators to density operators"))
    end
    bl, br = choi.basis_l[1], choi.basis_r[1]
    #ishermitian(choi.data) || @warn "ChoiState is not hermitian"
    # TODO: figure out how to do this with sparse matrices using e.g. Arpack.jl or ArnoldiMethod.jl
    vals, vecs = eigen(Hermitian(Matrix(choi.data)))
    for val in vals
        (abs(val) > tol && val < 0) && @warn "eigval $(val) < 0 but abs(eigval) > tol=$(tol)"
    end
    ops = [Operator(bl, br, sqrt(val)*reshape(vecs[:,i], length(bl), length(br)))
           for (i, val) in enumerate(vals) if abs(val) > tol && val > 0]
    return KrausOperators(bl, br, ops)
end

KrausOperators(op::SuperOperator; tol=1e-9) = KrausOperators(ChoiState(op; tol=tol); tol=tol)
