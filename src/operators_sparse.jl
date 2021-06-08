import Base: ==, *, /, +, -, Broadcast
import SparseArrays: sparse

const SparseOpPureType{BL,BR} = Operator{BL,BR,<:SparseMatrixCSC}
const SparseOpAdjType{BL,BR} = Operator{BL,BR,<:Adjoint{<:Number,<:SparseMatrixCSC}}
const SparseOpType{BL,BR} = Union{SparseOpPureType{BL,BR},SparseOpAdjType{BL,BR}}


"""
    SparseOperator(b1[, b2, data])

Sparse array implementation of Operator.

The matrix is stored as the julia built-in type `SparseMatrixCSC`
in the `data` field.
"""
SparseOperator(b1::Basis, b2::Basis, data) = Operator(b1, b2, SparseMatrixCSC(data))
SparseOperator(b1::Basis, b2::Basis, data::SparseMatrixCSC) = Operator(b1, b2, data)
SparseOperator(b::Basis, data) = SparseOperator(b, b, data)
SparseOperator(op::DataOperator) = SparseOperator(op.basis_l, op.basis_r, op.data)

SparseOperator(::Type{T},b1::Basis,b2::Basis) where T = SparseOperator(b1,b2,spzeros(T,length(b1),length(b2)))
SparseOperator(::Type{T},b::Basis) where T = SparseOperator(b,b,spzeros(T,length(b),length(b)))
SparseOperator(b1::Basis, b2::Basis) = SparseOperator(ComplexF64, b1, b2)
SparseOperator(b::Basis) = SparseOperator(ComplexF64, b, b)

"""
    sparse(op::AbstractOperator)

Convert an arbitrary operator into a [`SparseOperator`](@ref).
"""
sparse(a::AbstractOperator) = throw(ArgumentError("Direct conversion from $(typeof(a)) not implemented. Use sparse(full(op)) instead."))
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

function permutesystems(rho::SparseOpPureType{B1,B2}, perm) where {B1<:CompositeBasis,B2<:CompositeBasis}
    @assert length(rho.basis_l.bases) == length(rho.basis_r.bases) == length(perm)
    @assert isperm(perm)
    shape = [rho.basis_l.shape; rho.basis_r.shape]
    data = permutedims(rho.data, shape, [perm; perm .+ length(perm)])
    SparseOperator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:DataOperator} = SparseOperator(b1, b2, sparse(ComplexF64(1)*I, length(b1), length(b2)))
identityoperator(b1::Basis, b2::Basis) = identityoperator(DataOperator, b1, b2)
identityoperator(b::Basis) = identityoperator(b, b)

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
