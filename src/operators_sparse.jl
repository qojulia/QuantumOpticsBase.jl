import Base: ==, *, /, +, -, Broadcast
import SparseArrays: sparse
import FastExpm: fastExpm

const SparseOpPureType{BL,BR} = Operator{BL,BR,<:SparseMatrixCSC}
const SparseOpAdjType{BL,BR} = Operator{BL,BR,<:Adjoint{<:Number,<:SparseMatrixCSC}}
const SparseOpType{BL,BR} = Union{SparseOpPureType{BL,BR},SparseOpAdjType{BL,BR}}

"""
    SparseOperator(b1[, b2, data])

Sparse array implementation of Operator.

The matrix is stored as the julia built-in type `SparseMatrixCSC` in the `data` field.
"""
SparseOperator(b1::Basis, b2::Basis, data) = Operator(b1, b2, SparseMatrixCSC(data))
SparseOperator(b1::Basis, b2::Basis, data::SparseMatrixCSC) = Operator(b1, b2, data)
SparseOperator(b::Basis, data) = SparseOperator(b, b, data)
SparseOperator(op::DataOperator) = SparseOperator(op.basis_l, op.basis_r, op.data)
SparseOperator(::Type{T},b1::Basis,b2::Basis) where T = SparseOperator(b1,b2,spzeros(T,length(b1),length(b2)))
SparseOperator(::Type{T},b::Basis) where T = SparseOperator(b,b,spzeros(T,length(b),length(b)))
SparseOperator(b1::Basis, b2::Basis) = SparseOperator(ComplexF64, b1, b2)
SparseOperator(b::Basis) = SparseOperator(ComplexF64, b, b)

sparse(a::DataOperator) = Operator(a.basis_l, a.basis_r, sparse(a.data))

function ptrace(op::SparseOpPureType, indices)
    check_ptrace_arguments(op, indices)
    shape = [op.basis_l.shape; op.basis_r.shape]
    data = ptrace(op.data, shape, indices)
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    Operator(b_l, b_r, data)
end

function expect(op::SparseOpPureType{B1,B2}, state::Operator{B2,B2}) where {B1,B2}
    check_samebases(op, state)
    result = zero(promote_type(eltype(op),eltype(state)))
    @inbounds for colindex = 1:op.data.n
        for i=op.data.colptr[colindex]:op.data.colptr[colindex+1]-1
            result += op.data.nzval[i]*state.data[colindex, op.data.rowval[i]]
        end
    end
    result
end

"""
    exp(op::SparseOpType; opts...)

Operator exponential used, for example, to calculate displacement operators.
Uses [`FastExpm.jl.jl`](https://github.com/fmentink/FastExpm.jl) which will return a sparse
or dense operator depending on which is more efficient.
All optional arguments are passed to `fastExpm` and can be used to specify tolerances.

If you only need the result of the exponential acting on a vector,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
function exp(op::T; opts...) where {B,T<:SparseOpType{B,B}}
    if iszero(op)
        return identityoperator(op)
    else
        return SparseOperator(op.basis_l, op.basis_r, fastExpm(op.data; opts...))
    end
end

identityoperator(::Type{T}, ::Type{S}, b1::Basis, b2::Basis) where {T<:SparseOpType,S<:Number} =
    SparseOperator(b1, b2, sparse(one(S)*I, length(b1), length(b2)))
identityoperator(::Type{T}, ::Type{S}, b::Basis) where {T<:SparseOpType,S<:Number} =
    SparseOperator(b, b, sparse(one(S)*I, length(b), length(b)))

const EyeOpPureType{BL,BR} = Operator{BL,BR,<:Eye}
const EyeOpAdjType{BL,BR} = Operator{BL,BR,<:Adjoint{<:Number,<:Eye}}
const EyeOpType{BL,BR} = Union{EyeOpPureType{BL,BR},EyeOpAdjType{BL,BR}}

const _EmbedOpType{BL,BR} = Union{DenseOpType{BL,BR},SparseOpType{BL,BR},EyeOpType{BL,BR}}

function _embed_strides(dimensions::Vector{Int})
    strides = Vector{Int}(undef, length(dimensions))
    total = 1
    for i in eachindex(dimensions)
        strides[i] = total
        total *= dimensions[i]
    end
    return strides, total
end

function _embed_offsets(dimensions::Vector{Int}, strides::Vector{Int}, indices::Vector{Int})
    offsets = Vector{Int}(undef, prod(dimensions))
    for linear_index in eachindex(offsets)
        coordinate = linear_index - 1
        offset = 0
        for i in eachindex(indices)
            coordinate, digit = divrem(coordinate, dimensions[i])
            offset += digit * strides[indices[i]]
        end
        offsets[linear_index] = offset
    end
    return offsets
end

function _embed_basis_match(
    full::CompositeBasis, indices::Vector{Int}, selected::B
) where B<:Basis
    candidate = reduced(full, indices)
    candidate isa B && return (candidate::B) == selected
    return (candidate == selected)::Bool
end

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
               indices::AbstractVector{<:Integer}, op::T) where T<:_EmbedOpType
    N = length(basis_l.bases)
    @assert length(basis_r.bases) == N
    check_indices(N, indices)

    indices_int = Int[index for index in indices]
    _embed_basis_match(basis_l, indices_int, op.basis_l) || throw(IncompatibleBases())
    _embed_basis_match(basis_r, indices_int, op.basis_r) || throw(IncompatibleBases())

    dimensions_l = Int[dimension for dimension in basis_l.shape]
    dimensions_r = Int[dimension for dimension in basis_r.shape]
    strides_l, total_l = _embed_strides(dimensions_l)
    strides_r, total_r = _embed_strides(dimensions_r)

    selected = falses(N)
    for index in indices_int
        selected[index] = true
    end
    complement_indices = Int[index for index in 1:N if !selected[index]]

    selected_rows = _embed_offsets(dimensions_l[indices_int], strides_l, indices_int)
    selected_columns = _embed_offsets(dimensions_r[indices_int], strides_r, indices_int)
    identity_dimensions = min.(dimensions_l[complement_indices], dimensions_r[complement_indices])
    identity_rows = _embed_offsets(identity_dimensions, strides_l, complement_indices)
    identity_columns = _embed_offsets(identity_dimensions, strides_r, complement_indices)

    data = SparseMatrixCSC{eltype(op),Int}(op.data)
    entry_count = count(!iszero, data.nzval) * length(identity_rows)
    rows = Vector{Int}(undef, entry_count)
    columns = Vector{Int}(undef, entry_count)
    values = Vector{eltype(op)}(undef, entry_count)

    entry = 1
    for column in axes(data, 2)
        for position in nzrange(data, column)
            value = data.nzval[position]
            iszero(value) && continue
            row_offset = selected_rows[data.rowval[position]]
            column_offset = selected_columns[column]
            for identity_index in eachindex(identity_rows)
                rows[entry] = row_offset + identity_rows[identity_index] + 1
                columns[entry] = column_offset + identity_columns[identity_index] + 1
                values[entry] = value
                entry += 1
            end
        end
    end

    return Operator(basis_l, basis_r, sparse(rows, columns, values, total_l, total_r))
end

identityoperator(::Type{T}, ::Type{S}, b1::Basis, b2::Basis) where {T<:DataOperator,S<:Number} =
    Operator(b1, b2, Eye{S}(length(b1), length(b2)))
identityoperator(::Type{T}, ::Type{S}, b::Basis) where {T<:DataOperator,S<:Number} =
    Operator(b, b, Eye{S}(length(b)))

identityoperator(::Type{T}, b1::Basis, b2::Basis) where T<:Number = identityoperator(DataOperator, T, b1, b2) # XXX This is purposeful type piracy over QuantumInterface, that hardcodes the use of QuantumOpticsBase.DataOperator in identityoperator. Also necessary for backward compatibility.
identityoperator(::Type{T}, b::Basis) where T<:Number = identityoperator(DataOperator, T, b)

"""
    diagonaloperator(b::Basis)

Create a diagonal operator of type [`SparseOperator`](@ref).
"""
function diagonaloperator(b::Basis, diag)
  @assert 1 <= length(diag) <= length(b)
  SparseOperator(b, spdiagm(0=>diag))
end

# Fast in-place multiplication implementations
mul!(result::DenseOpType{B1,B3},M::SparseOpType{B1,B2},b::DenseOpType{B2,B3},alpha,beta) where {B1,B2,B3} = (gemm!(alpha,M.data,b.data,beta,result.data); result)
mul!(result::DenseOpType{B1,B3},a::DenseOpType{B1,B2},M::SparseOpType{B2,B3},alpha,beta) where {B1,B2,B3} = (gemm!(alpha,a.data,M.data,beta,result.data); result)
mul!(result::Ket{B1},M::SparseOpPureType{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2} = (gemv!(alpha,M.data,b.data,beta,result.data); result)
mul!(result::Bra{B2},b::Bra{B1},M::SparseOpPureType{B1,B2},alpha,beta) where {B1,B2} = (gemv!(alpha,b.data,M.data,beta,result.data); result)

# Ensure that Eye is not densified # TODO - some of this can still be special cased on Eye or lazy embed
+(op1::EyeOpType{BL,BR},op2::SparseOpType{BL,BR}) where {BL,BR} = sparse(op1) + op2
-(op1::EyeOpType{BL,BR},op2::SparseOpType{BL,BR}) where {BL,BR} = sparse(op1) - op2
+(op1::SparseOpType{BL,BR},op2::EyeOpType{BL,BR}) where {BL,BR} = sparse(op2) + op1
-(op2::SparseOpType{BL,BR},op1::EyeOpType{BL,BR}) where {BL,BR} = op2 - sparse(op1)
+(op1::EyeOpType{BL,BR},op2::EyeOpType{BL,BR}) where {BL,BR} = sparse(op1) + sparse(op2)
-(op2::EyeOpType{BL,BR},op1::EyeOpType{BL,BR}) where {BL,BR} = sparse(op2) - sparse(op1)
-(op::EyeOpType) = -sparse(op)
*(op::EyeOpType,a::T) where {T<:Number} = a*sparse(op)
*(a::T,op::EyeOpType) where {T<:Number} = a*sparse(op)
/(op::EyeOpType,a::T) where {T<:Number} = sparse(op)/a
tensor(a::EyeOpType, b::SparseOpType) = tensor(sparse(a),b)
tensor(a::SparseOpType, b::EyeOpType) = tensor(a,sparse(b))
tensor(a::EyeOpType, b::EyeOpType) = tensor(sparse(a),sparse(b))
dagger(a::EyeOpType) = dagger(sparse(a))
