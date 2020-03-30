import Base: permutedims

const SparseMatrix = SparseMatrixCSC{ComplexF64, Int}

# Simple mul! for Bra*SparseOperator
function mul!(result::Vector{ComplexF64},v::Vector{ComplexF64},M::SparseMatrix,alpha,beta)
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
    y = spzeros(ComplexF64, shape_2d_after...)
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
    y = spzeros(ComplexF64, x.m, x.n)
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
