import Base: permutedims

function gemm_sp_dense_small(alpha, M::SparseMatrixCSC, B::Matrix, result::Matrix)
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

function gemm_sp_dense_big(alpha, M::SparseMatrixCSC, B::Matrix, result::Matrix)
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

function gemm!(alpha, M::SparseMatrixCSC, B::Matrix, beta, result::Matrix)
    if iszero(beta)
        fill!(result, beta)
    elseif !isone(beta)
        rmul!(result, beta)
    end
    if nnz(M) > 1000
        gemm_sp_dense_big(alpha, M, B, result)
    else
        gemm_sp_dense_small(alpha, M, B, result)
    end
end

function gemm!(alpha, B::Matrix, M::SparseMatrixCSC, beta, result::Matrix)
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

function gemv!(alpha, M::SparseMatrixCSC, v::Vector, beta, result::Vector)
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

function gemv!(alpha, v::Vector, M::SparseMatrixCSC, beta, result::Vector)
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

function sub2sub(shape1::NTuple{N, Int}, shape2::NTuple{M, Int}, I::CartesianIndex{N}) where {N, M}
    linearindex = LinearIndices(shape1)[I.I...]
    CartesianIndices(shape2)[linearindex]
end

function ptrace(x, shape_nd::Vector{Int}, indices::Vector{Int})
    shape_nd = (shape_nd...,)
    N = div(length(shape_nd), 2)
    shape_2d = (x.m, x.n)
    shape_nd_after = ([i ∈ indices || i-N ∈ indices ? 1 : shape_nd[i] for i=1:2*N]...,)
    shape_2d_after = (prod(shape_nd_after[1:N]), prod(shape_nd_after[N+1:end]))
    I_nd_after_max = CartesianIndex(shape_nd_after...)
    y = spzeros(eltype(x), shape_2d_after...)
    for I in eachindex(x)
        I_nd = sub2sub(shape_2d, shape_nd, I)
        if I_nd.I[indices] != I_nd.I[indices .+ N]
            continue
        end
        I_after = sub2sub(shape_nd_after, shape_2d_after, min(I_nd, I_nd_after_max))
        y[I_after] += x[I]
    end
    y
end

function permutedims(x, shape, perm)
    shape = (shape...,)
    shape_perm = ([shape[i] for i in perm]...,)
    y = spzeros(eltype(x), x.m, x.n)
    for I in eachindex(x)
        linear_index = LinearIndices((x.m, x.n))[I.I...]
        cartesian_index = CartesianIndices(shape)[linear_index]
        cartesian_index_perm = [cartesian_index[i] for i=perm]
        linear_index_perm = LinearIndices(shape_perm)[cartesian_index_perm...]
        J = Tuple(CartesianIndices((x.m, x.n))[linear_index_perm])
        y[J...] = x[I.I...]
    end
    y
end
