import Base: ==, +, -, *, /, Broadcast
import Adapt
using Base.Cartesian

"""
    Operator{BL,BR,T} <: DataOperator{BL,BR}

Operator type that stores the representation of an operator on the Hilbert spaces
given by `BL` and `BR` (e.g. a Matrix).
"""
mutable struct Operator{BL,BR,T} <: DataOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::T
    function Operator{BL,BR,T}(basis_l::BL,basis_r::BR,data::T) where {BL,BR,T}
        (length.((basis_l,basis_r))==size(data)) || throw(DimensionMismatch("Tried to assign data of size $(size(data)) to bases of length $(length(basis_l)) and $(length(basis_r))!"))
        new(basis_l,basis_r,data)
    end
end
Operator{BL,BR}(basis_l::BL,basis_r::BR,data::T) where {BL,BR,T} = Operator{BL,BR,T}(basis_l,basis_r,data)
Operator(basis_l::BL,basis_r::BR,data::T) where {BL,BR,T} = Operator{BL,BR,T}(basis_l,basis_r,data)
Operator(b::Basis,data) = Operator(b,b,data)

Base.zero(op::Operator) = Operator(op.basis_l,op.basis_r,zero(op.data))
Base.eltype(op::Operator) = eltype(op.data)
Base.size(op::Operator) = size(op.data)
Base.size(op::Operator, d::Int) = size(op.data, d)

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
DenseOperator(::Type{T},b1::Basis,b2::Basis) where T = Operator(b1,b2,zeros(T,length(b1),length(b2)))
DenseOperator(::Type{T},b::Basis) where T = Operator(b,b,zeros(T,length(b),length(b)))
DenseOperator(b1::Basis, b2::Basis) = DenseOperator(ComplexF64, b1, b2)
DenseOperator(b::Basis) = DenseOperator(ComplexF64, b)
DenseOperator(op::DataOperator) = DenseOperator(op.basis_l,op.basis_r,Matrix(op.data))

Base.copy(x::Operator) = Operator(x.basis_l, x.basis_r, copy(x.data))

"""
    dense(op::AbstractOperator)

Convert an arbitrary Operator into a [`DenseOperator`](@ref).
"""
dense(x::AbstractOperator) = DenseOperator(x)

==(x::DataOperator{BL,BR}, y::DataOperator{BL,BR}) where {BL,BR} = (samebases(x,y) && x.data==y.data)
==(x::DataOperator, y::DataOperator) = false
Base.isapprox(x::DataOperator{BL,BR}, y::DataOperator{BL,BR}; kwargs...) where {BL,BR} = (samebases(x,y) && isapprox(x.data, y.data; kwargs...))
Base.isapprox(x::DataOperator, y::DataOperator; kwargs...) = false

# Arithmetic operations
+(a::Operator{BL,BR}, b::Operator{BL,BR}) where {BL,BR} = Operator(a.basis_l, a.basis_r, a.data+b.data)
+(a::Operator, b::Operator) = throw(IncompatibleBases())

-(a::Operator) = Operator(a.basis_l, a.basis_r, -a.data)
-(a::Operator{BL,BR}, b::Operator{BL,BR}) where {BL,BR} = Operator(a.basis_l, a.basis_r, a.data-b.data)
-(a::Operator, b::Operator) = throw(IncompatibleBases())

*(a::Operator{BL,BR}, b::Ket{BR}) where {BL,BR} = Ket{BL}(a.basis_l, a.data*b.data)
*(a::DataOperator, b::Ket) = throw(IncompatibleBases())
*(a::Bra{BL}, b::Operator{BL,BR}) where {BL,BR} = Bra{BR}(b.basis_r, transpose(b.data)*a.data)
*(a::Bra, b::DataOperator) = throw(IncompatibleBases())
*(a::Operator{B1,B2}, b::Operator{B2,B3}) where {B1,B2,B3} = Operator(a.basis_l, b.basis_r, a.data*b.data)
*(a::DataOperator, b::DataOperator) = throw(IncompatibleBases())
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, b*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, a*b.data)
function *(op1::AbstractOperator{B1,B2}, op2::Operator{B2,B3,T}) where {B1,B2,B3,T}
    result = Operator{B1,B3,T}(op1.basis_l, op2.basis_r, similar(op2.data,length(op1.basis_l),length(op2.basis_r)))
    mul!(result,op1,op2)
    return result
end
function *(op1::Operator{B1,B2,T}, op2::AbstractOperator{B2,B3}) where {B1,B2,B3,T}
    result = Operator{B1,B3,T}(op1.basis_l, op2.basis_r, similar(op1.data,length(op1.basis_l),length(op2.basis_r)))
    mul!(result,op1,op2)
    return result
end
function *(op::AbstractOperator{BL,BR}, psi::Ket{BR,T}) where {BL,BR,T}
    result = Ket{BL,T}(op.basis_l,similar(psi.data,length(op.basis_l)))
    mul!(result,op,psi)
    return result
end
function *(psi::Bra{BL,T}, op::AbstractOperator{BL,BR}) where {BL,BR,T}
    result = Bra{BR,T}(op.basis_r, similar(psi.data,length(op.basis_r)))
    mul!(result,psi,op)
    return result
end

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
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(b.data, a.data), length(a.basis), length(b.basis)))

tr(op::Operator{B,B}) where B = tr(op.data)

function ptrace(a::DataOperator, indices)
    check_ptrace_arguments(a, indices)
    rank = length(a.basis_l.shape)
    result = _ptrace(Val{rank}, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end
ptrace(op::AdjointOperator, indices) = dagger(ptrace(op, indices))

function ptrace(psi::Ket, indices)
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_ket(Val{rank}, psi.data, b.shape, indices)::Matrix{eltype(psi)}
    return Operator(b_, b_, result)
end

function ptrace(psi::Bra, indices)
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_bra(Val{rank}, psi.data, b.shape, indices)::Matrix{eltype(psi)}
    return Operator(b_, b_, result)
end

normalize!(op::Operator) = (rmul!(op.data, 1.0/tr(op)); op)

function expect(op::DataOperator{B,B}, state::Ket{B}) where B
    state.data' * op.data * state.data
end

function expect(op::DataOperator{B1,B2}, state::DataOperator{B2,B2}) where {B1,B2}
    check_samebases(op, state)
    result = zero(promote_type(eltype(op),eltype(state)))
    @inbounds for i=1:size(op.data, 1), j=1:size(op.data,2)
        result += op.data[i,j]*state.data[j,i]
    end
    result
end

function exp(op::T) where {B,T<:DenseOpType{B,B}}
    return DenseOperator(op.basis_l, op.basis_r, exp(op.data))
end

function permutesystems(a::Operator{B1,B2}, perm) where {B1<:CompositeBasis,B2<:CompositeBasis}
    @assert length(a.basis_l.bases) == length(a.basis_r.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(a.data, [a.basis_l.shape; a.basis_r.shape]...)
    data = permutedims(data, [perm; perm .+ length(perm)])
    data = reshape(data, length(a.basis_l), length(a.basis_r))
    return Operator(permutesystems(a.basis_l, perm), permutesystems(a.basis_r, perm), data)
end
permutesystems(a::AdjointOperator{B1,B2}, perm) where {B1<:CompositeBasis,B2<:CompositeBasis} = dagger(permutesystems(dagger(a),perm))

identityoperator(::Type{T}, b1::Basis, b2::Basis) where {BL,BR,dType,T<:DenseOpType} = Operator(b1, b2, Matrix{ComplexF64}(I, length(b1), length(b2)))

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
mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3} = (LinearAlgebra.mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Ket{B1},a::Operator{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2} = (LinearAlgebra.mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Bra{B2},a::Bra{B1},b::Operator{B1,B2},alpha,beta) where {B1,B2} = (LinearAlgebra.mul!(result.data,transpose(b.data),a.data,alpha,beta); result)
rmul!(op::Operator, x) = (rmul!(op.data, x); op)

# Multiplication for Operators in terms of their gemv! implementation
function mul!(result::Operator{B1,B3},M::AbstractOperator{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3}
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        mul!(resultket,M,bket,alpha,beta)
        result.data[:,i] = resultket.data
    end
    return result
end

function mul!(result::Operator{B1,B3},b::Operator{B1,B2},M::AbstractOperator{B2,B3},alpha,beta) where {B1,B2,B3}
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
@inline Base.axes(A::DataOperator) = axes(A.data)
Base.broadcastable(A::DataOperator) = A

# Custom broadcasting styles
abstract type DataOperatorStyle{BL,BR} <: Broadcast.BroadcastStyle end
struct OperatorStyle{BL,BR} <: DataOperatorStyle{BL,BR} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:Operator{BL,BR}}) where {BL,BR} = OperatorStyle{BL,BR}()
Broadcast.BroadcastStyle(::OperatorStyle{B1,B2}, ::OperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:OperatorStyle{BL,BR},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    return Operator{BL,BR}(bl, br, copy(bc_))
end
find_basis(a::DataOperator, rest) = (a.basis_l, a.basis_r)

const BasicMathFunc = Union{typeof(+),typeof(-),typeof(*)}
function Broadcasted_restrict_f(f::BasicMathFunc, args::Tuple{Vararg{<:DataOperator}}, axes)
    args_ = Tuple(a.data for a=args)
    return Broadcast.Broadcasted(f, args_, axes)
end
function Broadcasted_restrict_f(f, args::Tuple{Vararg{<:DataOperator}}, axes)
    throw(error("Cannot broadcast function `$f` on type `$(eltype(args))`"))
end

# In-place broadcasting
@inline function Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:DataOperatorStyle{BL,BR},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && isa(bc.args, Tuple{<:DataOperator{BL,BR}}) # only a single input argument to broadcast!
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
@inline Base.copyto!(A::DataOperator{BL,BR},B::DataOperator{BL,BR}) where {BL,BR} = (copyto!(A.data,B.data); A)
@inline Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:DataOperatorStyle,Axes,F,Args} =
    throw(IncompatibleBases())
