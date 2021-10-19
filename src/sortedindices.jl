"""
6, [1, 4] => [2, 3, 5, 6]
"""
function complement(N, indices)
    L = length(indices)
    x = Vector{Int}(undef, N - L)
    i_ = 1 # Position in the x vector
    j = 1 # Position in indices vector
    for i=1:N
        if j > L || indices[j]!=i
            x[i_] = i
            i_ += 1
        else
            j += 1
        end
    end
    x
end


"""
[1, 4, 5], [2, 4, 7] => [1, 5]
"""
function remove(ind1, ind2)
    x = Int[]
    for i in ind1
        if i ∉ ind2
            push!(x, i)
        end
    end
    x
end

"""
[1, 4, 5], [2, 4, 7] => [1, 3]
"""
function shiftremove(ind1, ind2)
    x = Int[]
    for i in ind1
        if i ∉ ind2
            counter = 0
            for i2 in ind2
                if i2 < i
                    counter += 1
                else
                    break
                end
            end
            push!(x, i-counter)
        end
    end
    x
end

function reducedindices(I_, I)
    N = length(I_)
    x = Vector{Int}(undef, N)
    for n in 1:N
        x[n] = findfirst(isequal(I_[n]), I)
    end
    x
end

function reducedindices!(I_, I)
    for n in 1:length(I_)
        I_[n] = findfirst(isequal(I_[n]), I)
    end
end

"""
Check if all indices are unique and smaller than or equal to imax.
"""
function check_indices(imax, indices)
    N = length(indices)
    for n=1:N
        i = indices[n]
        @assert 0 < i <= imax
        for m in n+1:N
            @assert i != indices[m]
        end
    end
end

"""
Check if the indices are sorted, unique and smaller than or equal to imax.
"""
function check_sortedindices(imax, indices)
    N = length(indices)
    if N == 0
        return nothing
    end
    i_ = indices[1]
    @assert 0 < i_ <= imax
    for i in indices[2:end]
        @assert 0 < i <= imax
        @assert i > i_
    end
end

"""
    check_embed_indices(indices::Array)

Determine whether a collection of indices, written as a list of (integers or lists of integers) is unique.
This assures that the embedded operators are in non-overlapping subspaces.
"""
function check_embed_indices(indices)
    # short circuit return when `indices` is empty.
    length(indices) == 0 && return true

    err_str = "Variable `indices` comes in an unexpected form. Expecting `Array{Union{Int, Array{Int, 1}}, 1}`"
    @assert all(x isa Array || x isa Int for x in indices) err_str

    # flatten the indices and check for uniqueness
    # use a custom flatten because it's ≈ 4x  faster than Base.Iterators.flatten
    flatten(arr) = mapreduce(x -> x isa Vector ? flatten(x) : x, append!, arr, init=[])
    allunique(flatten(indices))
end
