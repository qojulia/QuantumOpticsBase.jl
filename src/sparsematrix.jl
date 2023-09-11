import Base: permutedims

function gemm_sp_dense_small(alpha, M::SparseMatrixCSC, B::AbstractMatrix, result::AbstractMatrix)
    if isone(alpha)
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                row = M.rowval[i]
                val = M.nzval[i]
                @inbounds for j=1:size(B, 2)
                    result[row, j] += val*B[colindex, j]
                end
            end
        end
    else
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                row = M.rowval[i]
                val = alpha*M.nzval[i]
                @inbounds for j=1:size(B, 2)
                    result[row, j] += val*B[colindex, j]
                end
            end
        end
    end
end

function gemm_sp_dense_big(alpha, M::SparseMatrixCSC, B::AbstractMatrix, result::AbstractMatrix)
    if isone(alpha)
        @inbounds for j=1:size(B, 2)
            @inbounds for colindex = 1:M.n
                m2 = B[colindex, j]
                @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                    row = M.rowval[i]
                    result[row, j] += M.nzval[i]*m2
                end
            end
        end
    else
        @inbounds for j=1:size(B, 2)
            @inbounds for colindex = 1:M.n
                m2 = alpha*B[colindex, j]
                @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                    row = M.rowval[i]
                    result[row, j] += M.nzval[i]*m2
                end
            end
        end
    end
end

function gemm_dense_adj_sp(alpha, B::AbstractMatrix, M::SparseMatrixCSC, result::AbstractMatrix)
    if isone(alpha)
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                row = M.rowval[i]
                val = conj(M.nzval[i])
                @inbounds for j=1:size(B, 1)
                    result[j, row] += val*B[j, colindex]
                end
            end
        end
    else
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                row = M.rowval[i]
                val = alpha*conj(M.nzval[i])
                @inbounds for j=1:size(B, 1)
                    result[j, row] += val*B[j, colindex]
                end
            end
        end
    end
end

function gemm_adj_sp_dense_small(alpha, M::SparseMatrixCSC, B::AbstractMatrix, result::AbstractMatrix)
    dimB = size(result,2)
    if isone(alpha)
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                mi = conj(M.nzval[i])
                mrowvali = M.rowval[i]
                @inbounds for j=1:dimB
                    result[colindex, j] += mi*B[mrowvali, j]
                end
            end
        end
    else
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                mi = conj(M.nzval[i])*alpha
                mrowvali = M.rowval[i]
                @inbounds for j=1:dimB
                    result[colindex, j] += mi*B[mrowvali, j]
                end
            end
        end
    end
end


function gemm!(alpha, M::SparseMatrixCSC, B::AbstractMatrix, beta, result::AbstractMatrix)
    size(M, 2) == size(B, 1) || throw(DimensionMismatch())
    size(M, 1) == size(result, 1) || throw(DimensionMismatch())
    size(B, 2) == size(result, 2) || throw(DimensionMismatch())
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    if nnz(M) > 550
        gemm_sp_dense_big(alpha, M, B, result)
    else
        gemm_sp_dense_small(alpha, M, B, result)
    end
end

function gemm!(alpha, B::AbstractMatrix, M::SparseMatrixCSC, beta, result::AbstractMatrix)
    size(M, 1) == size(B, 2) || throw(DimensionMismatch())
    size(M, 2) == size(result,2) || throw(DimensionMismatch())
    size(B, 1) == size(result,1) || throw(DimensionMismatch())
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    dimB = size(result,1)
    if isone(alpha)
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                mi = M.nzval[i]
                mrowvali = M.rowval[i]
                @inbounds for j=1:dimB
                    result[j, colindex] += mi*B[j, mrowvali]
                end
            end
        end
    else
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                mi = M.nzval[i]*alpha
                mrowvali = M.rowval[i]
                @inbounds for j=1:dimB
                    result[j, colindex] += mi*B[j, mrowvali]
                end
            end
        end
    end
end

function gemm!(alpha, M_::Adjoint{T,<:SparseMatrixCSC{T}}, B::AbstractMatrix, beta, result::AbstractMatrix) where T
    M = M_.parent
    if nnz(M) > 550
        LinearAlgebra.mul!(result, M_, B, alpha, beta)
    else
        size(M_, 2) == size(B, 1) || throw(DimensionMismatch())
        size(M_, 1) == size(result, 1) || throw(DimensionMismatch())
        size(B, 2) == size(result, 2) || throw(DimensionMismatch())
        if iszero(beta)
            fill!(result, beta)
        elseif !isone(beta)
            rmul!(result, beta)
        end
        gemm_adj_sp_dense_small(alpha, M, B, result)
    end
end

function gemm!(alpha, B::AbstractMatrix, M::Adjoint{T,<:SparseMatrixCSC{T}}, beta, result::AbstractMatrix) where T
    size(M, 1) == size(B, 2) || throw(DimensionMismatch())
    size(M, 2) == size(result,2) || throw(DimensionMismatch())
    size(B, 1) == size(result,1) || throw(DimensionMismatch())
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    gemm_dense_adj_sp(alpha, B, M.parent, result)
end

function gemm!(alpha, A::Adjoint{T, <:SparseMatrixCSC{T}}, B::Adjoint{S, <:SparseMatrixCSC{S}}, beta, result::AbstractMatrix) where {T,S}
    error("Matrix multiplication between Adjoint{SparseCSC} and SparseCSC matrices is not implemented yet. Submit an issue to the developers.")
end
function gemm!(alpha, A::Adjoint{T, <:SparseMatrixCSC{T}}, B::SparseMatrixCSC, beta, result::AbstractMatrix) where T
    error("Matrix multiplication between Adjoint{SparseCSC} and Adjoint{SparseCSC} matrices is not implemented yet. Submit an issue to the developers.")
end
function gemm!(alpha, A::SparseMatrixCSC, B::Adjoint{T, <:SparseMatrixCSC{T}}, beta, result::AbstractMatrix) where T
    error("Matrix multiplication between SparseCSC and Adjoint{SparseCSC} matrices is not implemented yet. Submit an issue to the developers.")
end
function gemm!(alpha, A::SparseMatrixCSC, B::SparseMatrixCSC, beta, result::AbstractMatrix)
    error("Matrix multiplication between SparseCSC and SparseCSC matrices is not implemented yet. Submit an issue to the developers.")
end

function gemv!(alpha, M::SparseMatrixCSC, v::AbstractVector, beta, result::AbstractVector)
    size(M, 2) == size(v, 1) || throw(DimensionMismatch())
    size(M, 1) == size(result, 1) || throw(DimensionMismatch())

    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    if isone(alpha)
        @inbounds for colindex = 1:M.n
            vj = v[colindex]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i]] += M.nzval[i]*vj
            end
        end
    else
        @inbounds for colindex = 1:M.n
            vj = alpha*v[colindex]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i]] += M.nzval[i]*vj
            end
        end
    end
end

function gemv!(alpha, v::AbstractVector, M::SparseMatrixCSC, beta, result::AbstractVector)
    size(M, 1) == size(v, 1) || throw(DimensionMismatch())
    size(M, 2) == size(result, 1) || throw(DimensionMismatch())

    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    if isone(alpha)
        @inbounds for colindex=1:M.n
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[colindex] += M.nzval[i]*v[M.rowval[i]]
            end
        end
    else
        @inbounds for colindex=1:M.n
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[colindex] += M.nzval[i]*alpha*v[M.rowval[i]]
            end
        end
    end
end




function _permutedims(x::AbstractSparseMatrix, shape, perm) # TODO upstream as `permutedims` to SparseArrays to avoid piracy -- used in a single location in operators_sparse
    shape = (shape...,)
    shape_perm = ([shape[i] for i in perm]...,)
    y = spzeros(eltype(x), x.m, x.n)
    for I in eachindex(x)::CartesianIndices # Help with inference (detected by JET)
        linear_index = LinearIndices((x.m, x.n))[I.I...]
        cartesian_index = CartesianIndices(shape)[linear_index]
        cartesian_index_perm = [cartesian_index[i] for i=perm]
        linear_index_perm = LinearIndices(shape_perm)[cartesian_index_perm...]
        J = Tuple(CartesianIndices((x.m, x.n))[linear_index_perm])
        y[J...] = x[I.I...]
    end
    y
end
