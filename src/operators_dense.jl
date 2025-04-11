import QuantumInterface
import Base: isequal, ==, +, -, *, /, Broadcast
import Adapt
using Base.Cartesian

"""
    Operator{BL,BR,T} <: DataOperator{BL,BR}

Operator type that stores the representation of an operator on the Hilbert spaces
given by `.basis_l` and `.basis_r` (e.g. a Matrix).
"""
mutable struct Operator{BL,BR,T} <: DataOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::T
    function Operator{BL,BR,T}(basis_l::BL,basis_r::BR,data::T) where {BL,BR,T}
        (dimension.((basis_l,basis_r))==size(data)) || throw(DimensionMismatch("Tried to assign data of size $(size(data)) to bases of length $(dimension(basis_l)) and $(dimension(basis_r))!"))
        new(basis_l,basis_r,data)
    end
end
Operator{BL,BR}(basis_l::BL,basis_r::BR,data::T) where {BL,BR,T} = Operator{BL,BR,T}(basis_l,basis_r,data)
Operator(basis_l::BL,basis_r::BR,data::T) where {BL,BR,T} = Operator{BL,BR,T}(basis_l,basis_r,data)
Operator(b::Basis,data) = Operator(b,b,data)
Operator(qet1::Ket, qetva::Ket...) = Operator(qet1.basis, GenericBasis(length(qetva)+1), qet1, qetva...)
Operator(basis_r::Basis,qet1::Ket,qetva::Ket...) = Operator(qet1.basis, basis_r, qet1, qetva...)
Operator(basis_l::BL,basis_r::BR,qet1::Ket,qetva::Ket...) where {BL,BR} = Operator{BL,BR}(basis_l, basis_r, hcat(qet1.data, getfield.(qetva,:data)...))
Operator(qets::AbstractVector{<:Ket}) = Operator(first(qets).basis, GenericBasis(length(qets)), qets)
Operator(basis_r::Basis,qets::AbstractVector{<:Ket}) = Operator(first(qets).basis, basis_r, qets)
Operator(basis_l::BL,basis_r::BR,qets::AbstractVector{<:Ket}) where {BL,BR} = Operator{BL,BR}(basis_l, basis_r, reduce(hcat, getfield.(qets, :data)))

basis_l(op::Operator) = op.basis_l
basis_r(op::Operator) = op.basis_r

QuantumInterface.traceout!(s::QuantumOpticsBase.Operator, i) = QuantumInterface.ptrace(s,i)

Base.zero(op::Operator) = Operator(op.basis_l,op.basis_r,zero(op.data))
Base.eltype(op::Operator) = eltype(op.data)
Base.eltype(::Type{T}) where {BL,BR,D,T<:Operator{BL,BR,D}} = eltype(D)
Base.size(op::Operator) = size(op.data)
Base.size(op::Operator, d::Int) = size(op.data, d)
function Base.convert(::Type{Operator{BL,BR,T}}, op::Operator{BL,BR,S}) where {BL,BR,T,S}
    if T==S
        return op
    else
        return Operator{BL,BR,T}(op.basis_l, op.basis_r, convert(T, op.data))
    end
end

# Convert data to CuArray with cu(::Operator)
Adapt.adapt_structure(to, x::Operator) = Operator(x.basis_l, x.basis_r, Adapt.adapt(to, x.data))

const DenseOpPureType{BL,BR} = Operator{BL,BR,<:Matrix}
const DenseOpAdjType{BL,BR} = Operator{BL,BR,<:Adjoint{<:Number,<:Matrix}}
const DenseOpType{BL,BR} = Union{DenseOpPureType{BL,BR},DenseOpAdjType{BL,BR}}
const AdjointOperator{BL,BR} = Operator{BL,BR,<:Adjoint}

"""
    DenseOperator(b1[, b2, data])

Dense array implementation of Operator. Converts any given data to a dense `Matrix`.
"""
DenseOperator(basis_l::Basis,basis_r::Basis,data::T) where T = Operator(basis_l,basis_r,Matrix(data))
DenseOperator(basis_l::Basis,basis_r::Basis,data::Matrix) = Operator(basis_l,basis_r,data)
DenseOperator(b::Basis, data) = DenseOperator(b, b, data)
DenseOperator(::Type{T},b1::Basis,b2::Basis) where T = Operator(b1,b2,zeros(T,dimension(b1),dimension(b2)))
DenseOperator(::Type{T},b::Basis) where T = Operator(b,b,zeros(T,dimension(b),dimension(b)))
DenseOperator(b1::Basis, b2::Basis) = DenseOperator(ComplexF64, b1, b2)
DenseOperator(b::Basis) = DenseOperator(ComplexF64, b)
DenseOperator(op::DataOperator) = DenseOperator(op.basis_l,op.basis_r,Matrix(op.data))

Base.copy(x::Operator) = Operator(x.basis_l, x.basis_r, copy(x.data))

"""
    dense(op::AbstractOperator)

Convert an arbitrary Operator into a [`DenseOperator`](@ref).
"""
dense(x::AbstractOperator) = DenseOperator(x)

isequal(x::DataOperator, y::DataOperator) = (addible(x,y) && isequal(x.data, y.data))
==(x::DataOperator, y::DataOperator) = (addible(x,y) && x.data==y.data)
Base.isapprox(x::DataOperator, y::DataOperator; kwargs...) = (addible(x,y) && isapprox(x.data, y.data; kwargs...))

# Arithmetic operations
+(a::Operator, b::Operator) = (check_addible(a,b); Operator(a.basis_l, a.basis_r, a.data+b.data))

-(a::Operator) = Operator(a.basis_l, a.basis_r, -a.data)
-(a::Operator, b::Operator) = (check_addible(a,b); Operator(a.basis_l, a.basis_r, a.data-b.data))

*(a::Operator, b::Ket) = (check_multiplicable(a,b); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::Operator) = (check_multiplicable(a,b); Bra(b.basis_r, transpose(b.data)*a.data))
*(a::Operator, b::Operator) = (check_multiplicable(a,b); Operator(a.basis_l, b.basis_r, a.data*b.data))
*(a::DataOperator, b::Operator) = error("no `*` method defined for DataOperator subtype $(typeof(a))") # defined to avoid method ambiguity
*(a::Operator, b::DataOperator) = error("no `*` method defined for DataOperator subtype $(typeof(b))") # defined to avoid method ambiguity
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, b*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, a*b.data)
function *(op1::AbstractOperator, op2::Operator)
    check_multiplicable(op1,op2)
    result = Operator(basis_l(op1), basis_r(op2), similar(_parent(op2.data), promote_type(eltype(op1), eltype(op2)), dimension(basis_l(op1)),dimension(basis_r(op2))))
    mul!(result,op1,op2)
    return result
end
function *(op1::Operator, op2::AbstractOperator)
    check_multiplicable(op1,op2)
    result = Operator(basis_l(op1), basis_r(op2), similar(_parent(op1.data), promote_type(eltype(op1), eltype(op2)), dimension(basis_l(op1)), dimension(basis_r(op2))))
    mul!(result,op1,op2)
    return result
end
function *(op::AbstractOperator, psi::Ket)
    check_multiplicable(op,psi)
    result = Ket(basis_l(op), similar(psi.data, dimension(basis_l(op))))
    mul!(result,op,psi)
    return result
end
function *(psi::Bra, op::AbstractOperator)
    check_multiplicable(psi,op)
    result = Bra(basis_r(op), similar(psi.data, dimension(basis_r(op))))
    mul!(result,psi,op)
    return result
end

_parent(x::T, x_parent::T) where T = x
_parent(x, x_parent) = _parent(x_parent, parent(x_parent))
_parent(x) = _parent(x, parent(x))

/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data ./ b)

dagger(x::Operator) = Operator(x.basis_r,x.basis_l,adjoint(x.data))
transpose(x::Operator) = Operator(x.basis_r,x.basis_l,transpose(x.data))
ishermitian(A::DataOperator) = false
ishermitian(A::DataOperator{B,B}) where B = ishermitian(A.data)
Base.collect(A::Operator) = Operator(A.basis_l, A.basis_r, collect(A.data))

tensor(a::Operator, b::Operator) = Operator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))

conj(a::Operator) = Operator(a.basis_l, a.basis_r, conj(a.data))
conj!(a::Operator) = (conj!(a.data); a)


"""
    tensor(x::Ket, y::Bra)

Outer product ``|x⟩⟨y|`` of the given states.
"""
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(b.data, a.data), dimension(a.basis), dimension(b.basis)))

"""
    tensor(a::AbstractOperator, b::Bra)
    tensor(a::Bra, b::AbstractOperator)
    tensor(a::AbstractOperator, b::Ket)
    tensor(a::Ket, b::AbstractOperator)

Products of operators and state vectors ``a ⊗ <b|``. The result is an isometry
in case the operator is unitary and state is normalized.
"""
function tensor(a::AbstractOperator, b::Bra)
    # upgrade the bra to an operator that projects onto a dim-1 space
    # NOTE: copy() works around non-sparse-preserving kron in case b.data is a SparseVector.
    b_op = Operator(GenericBasis(1), basis(b), copy(reshape(b.data, (1,:))))
    ab_op = tensor(a, b_op)
    # squeeze out the trivial dimension
    Operator(a.basis_l, ab_op.basis_r, ab_op.data)
end

function tensor(a::Bra, b::AbstractOperator)
    # upgrade the bra to an operator that projects onto a dim-1 space
    a_op = Operator(GenericBasis(1), basis(a), copy(reshape(a.data, (1,:))))
    ab_op = tensor(a_op, b)
    # squeeze out the trivial dimension
    Operator(b.basis_l, ab_op.basis_r, ab_op.data)
end

function tensor(a::AbstractOperator, b::Ket)
    # upgrade the bra to an operator that projects onto a dim-1 space
    b_op = Operator(basis(b), GenericBasis(1), copy(reshape(b.data, (:,1))))
    ab_op = tensor(a, b_op)
    # squeeze out the trivial dimension
    Operator(ab_op.basis_l, a.basis_r, ab_op.data)
end

function tensor(a::Ket, b::AbstractOperator)
    # upgrade the bra to an operator that projects onto a dim-1 space
    a_op = Operator(basis(a), GenericBasis(1), copy(reshape(a.data, (:,1))))
    ab_op = tensor(a_op, b)
    # squeeze out the trivial dimension
    Operator(ab_op.basis_l, b.basis_r, ab_op.data)
end

tr(op::Operator{B,B}) where B = tr(op.data)

function ptrace(a::DataOperator, indices)
    check_ptrace_arguments(a, indices)
    rank = length(a.basis_l)
    result = _ptrace(Val{rank}, a.data, shape(a.basis_l), shape(a.basis_r), indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end
ptrace(op::AdjointOperator, indices) = dagger(ptrace(op, indices))

function ptrace(psi::Ket, indices)
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b)
    result = _ptrace_ket(Val{rank}, psi.data, shape(b), indices)::Matrix{eltype(psi)}
    return Operator(b_, b_, result)
end

function ptrace(psi::Bra, indices)
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b)
    result = _ptrace_bra(Val{rank}, psi.data, shape(b), indices)::Matrix{eltype(psi)}
    return Operator(b_, b_, result)
end

normalize!(op::Operator) = (rmul!(op.data, 1.0/tr(op)); op)

function expect(op::DataOperator, state::Ket)
    check_multiplicable(op,op); check_multiplicable(op, state)
    dot(state.data, op.data, state.data)
end

function expect(op::DataOperator, state::DataOperator)
    check_multiplicable(op,state); check_multiplicable(state, state)
    result = zero(promote_type(eltype(op),eltype(state)))
    @inbounds for i=1:size(op.data, 1), j=1:size(op.data,2)
        result += op.data[i,j]*state.data[j,i]
    end
    result
end

"""
    exp(op::DenseOpType)

Operator exponential used, for example, to calculate displacement operators.
Uses LinearAlgebra's `Base.exp`.

If you only need the result of the exponential acting on a vector,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
function exp(op::T) where {B,T<:DenseOpType{B,B}}
    return DenseOperator(op.basis_l, op.basis_r, exp(op.data))
end

function permutesystems(a::Operator{B1,B2}, perm) where {B1<:CompositeBasis,B2<:CompositeBasis}
    @assert length(a.basis_l) == length(a.basis_r) == length(perm)
    @assert isperm(perm)
    data = Base.ReshapedArray(a.data, (a.basis_l.shape..., a.basis_r.shape...), ())
    data = PermutedDimsArray(data, [perm; perm .+ length(perm)])
    data = Base.ReshapedArray(data, (dimension(a.basis_l), dimension(a.basis_r)), ())
    return Operator(permutesystems(a.basis_l, perm), permutesystems(a.basis_r, perm), copy(data))
end
permutesystems(a::AdjointOperator{B1,B2}, perm) where {B1<:CompositeBasis,B2<:CompositeBasis} = dagger(permutesystems(dagger(a),perm))

identityoperator(::Type{S}, ::Type{T}, b1::Basis, b2::Basis) where {S<:DenseOpType,T<:Number} =
    Operator(b1, b2, Matrix{T}(I, dimension(b1), dimension(b2)))

"""
    projector(a::Ket, b::Bra)

Projection operator ``|a⟩⟨b|``.
"""
projector(a::Ket, b::Bra) = tensor(a, b)
"""
    projector(a::Ket)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Ket) = Operator(a.basis, a.data*a.data')
"""
    projector(a::Bra)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Bra) = projector(a')

"""
    dm(a::StateVector)

Create density matrix ``|a⟩⟨a|``. Same as `projector(a)`.
"""
dm(x::Ket) = tensor(x, dagger(x))
dm(x::Bra) = tensor(dagger(x), x)


# Partial trace implementation for dense operators.
function _strides(shape)
    N = length(shape)
    S = zeros(eltype(shape), N)
    S[1] = 1
    for m=2:N
        S[m] = S[m-1]*shape[m-1]
    end
    return S
end

function _strides(shape::Ty)::Ty where Ty <: Tuple
    accumulate(*, (1,Base.front(shape)...))
end

# Dense operator version
@generated function _ptrace(::Type{Val{RANK}}, a,
                            shape_l, shape_r,
                            indices) where RANK
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = copy(shape_l)
        @inbounds for idx ∈ indices
            result_shape_l[idx] = 1
        end
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = copy(shape_r)
        @inbounds for idx ∈ indices
            result_shape_r[idx] = 1
        end
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(eltype(a), N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

@generated function _ptrace_ket(::Type{Val{RANK}}, a,
                            shape, indices) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        @inbounds for idx ∈ indices
            result_shape[idx] = 1
        end
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(eltype(a), N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0]*conj(a[Ir_0])
            end
        end
        return result
    end
end

@generated function _ptrace_bra(::Type{Val{RANK}}, a,
                            shape, indices) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        @inbounds for idx ∈ indices
            result_shape[idx] = 1
        end
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(eltype(a), N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += conj(a[Il_0])*a[Ir_0]
            end
        end
        return result
    end
end

"""
    mul!(Y::DataOperator,A::AbstractOperator,B::DataOperator,alpha,beta) -> Y
    mul!(Y::StateVector,A::AbstractOperator,B::StateVector,alpha,beta) -> Y

Fast in-place multiplication of operators/state vectors. Updates `Y` as
`Y = alpha*A*B + beta*Y`. In most cases, the call gets forwarded to
Julia's 5-arg mul! implementation on the underlying data.
See also [`LinearAlgebra.mul!`](@ref).
"""
mul!(result::Operator,a::Operator,b::Operator,alpha,beta) = (LinearAlgebra.mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Ket,a::Operator,b::Ket,alpha,beta) = (LinearAlgebra.mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Bra,a::Bra,b::Operator,alpha,beta) = (LinearAlgebra.mul!(result.data,transpose(b.data),a.data,alpha,beta); result)
rmul!(op::Operator, x) = (rmul!(op.data, x); op)

# Multiplication for Operators in terms of their gemv! implementation
function mul!(result::Operator,M::AbstractOperator,b::Operator,alpha,beta)
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        mul!(resultket,M,bket,alpha,beta)
        result.data[:,i] = resultket.data
    end
    return result
end

function mul!(result::Operator,b::Operator,M::AbstractOperator,alpha,beta)
    for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        mul!(resultbra,bbra,M,alpha,beta)
        result.data[i,:] = resultbra.data
    end
    return result
end

# Broadcasting
Base.size(A::DataOperator) = size(A.data)
Base.size(A::DataOperator, d) = size(A.data, d)
Base.size(A::DataOperator, d::Int) = size(A.data, d) # defined to avoid method ambiguity
@inline Base.axes(A::DataOperator) = axes(A.data)
Base.broadcastable(A::DataOperator) = A

# Custom broadcasting styles
abstract type DataOperatorStyle{BL,BR} <: Broadcast.BroadcastStyle end
struct OperatorStyle{BL,BR} <: DataOperatorStyle{BL,BR} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:Operator{BL,BR}}) where {BL,BR} = OperatorStyle{BL,BR}()
Broadcast.BroadcastStyle(::OperatorStyle{B1,B2}, ::OperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

# Broadcast with scalars (of use in ODE solvers checking for tolerances, e.g. `.* reltol .+ abstol`)
Broadcast.BroadcastStyle(::T, ::Broadcast.DefaultArrayStyle{0}) where {Bl<:Basis, Br<:Basis, T<:OperatorStyle{Bl,Br}} = T()

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:OperatorStyle{BL,BR},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    T = find_dType(bcf)
    data = zeros(T, dimension(bl), dimension(br))
    @inbounds @simd for I in eachindex(bcf)
        data[I] = bcf[I]
    end
    return Operator{BL,BR}(bl, br, data)
end

find_basis(a::DataOperator, rest) = (a.basis_l, a.basis_r)
find_dType(a::DataOperator, rest) = eltype(a)
@inline Base.getindex(a::DataOperator, idx) = getindex(a.data, idx)
Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(x::DataOperator, i) = x.data[i]
Base.iterate(a::DataOperator) = iterate(a.data)
Base.iterate(a::DataOperator, idx) = iterate(a.data, idx)

# In-place broadcasting
@inline function Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:DataOperatorStyle{BL,BR},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Base.Broadcast.preprocess(dest, bc)
    dest′ = dest.data
    @inbounds @simd for I in eachindex(bc′)
        dest′[I] = bc′[I]
    end
    return dest
end
@inline Base.copyto!(A::DataOperator{BL,BR},B::DataOperator{BL,BR}) where {BL,BR} = (copyto!(A.data,B.data); A)
@inline Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:DataOperatorStyle,Axes,F,Args} =
    throw(IncompatibleBases())

# A few more standard interfaces: These do not necessarily make sense for a StateVector, but enable transparent use of DifferentialEquations.jl
Base.eltype(::Type{Operator{Bl,Br,A}}) where {Bl,Br,N,A<:AbstractMatrix{N}} = N # ODE init
Base.any(f::Function, x::Operator; kwargs...) = any(f, x.data; kwargs...) # ODE nan checks
Base.all(f::Function, x::Operator; kwargs...) = all(f, x.data; kwargs...)
Base.fill!(x::Operator, a) = typeof(x)(x.basis_l, x.basis_r, fill!(x.data, a))
Base.ndims(x::Type{Operator{Bl,Br,A}}) where {Bl,Br,N,A<:AbstractMatrix{N}} = ndims(A)
Base.similar(x::Operator, t) = typeof(x)(x.basis_l, x.basis_r, copy(x.data))
RecursiveArrayTools.recursivecopy!(dest::Operator{Bl,Br,A},src::Operator{Bl,Br,A}) where {Bl,Br,A} = copyto!(dest,src) # ODE in-place equations
RecursiveArrayTools.recursivecopy(x::Operator) = copy(x)
RecursiveArrayTools.recursivecopy(x::AbstractArray{T}) where {T<:Operator} = copy(x)
RecursiveArrayTools.recursivefill!(x::Operator, a) = fill!(x, a)
