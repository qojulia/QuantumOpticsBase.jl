using Test
using QuantumInterface

@testset "sortedindices" begin

s = QuantumInterface

@test s.complement(6, [1, 4]) == [2, 3, 5, 6]

@test s.remove([1, 4, 5], [2, 4, 7]) == [1, 5]
@test s.remove([1, 4, 5, 7], [2, 4, 7]) == [1, 5]
@test s.remove([1, 4, 5, 8], [2, 4, 7]) == [1, 5, 8]

@test s.shiftremove([1, 4, 5], [2, 4, 7]) == [1, 3]
@test s.shiftremove([1, 4, 5, 7], [2, 4, 7]) == [1, 3]
@test s.shiftremove([1, 4, 5, 8], [2, 4, 7]) == [1, 3, 5]

@test s.reducedindices([3, 5], [2, 3, 5, 6]) == [2, 3]
x = [3, 5]
s.reducedindices!(x, [2, 3, 5, 6])
@test x == [2, 3]

@test_throws AssertionError s.check_indices(5, [1, 6])
@test_throws AssertionError s.check_indices(5, [0, 2])
@test s.check_indices(5, Int[]) == nothing
@test s.check_indices(5, [1, 3]) == nothing
@test s.check_indices(5, [3, 1]) == nothing

@test_throws AssertionError s.check_sortedindices(5, [1, 6])
@test_throws AssertionError s.check_sortedindices(5, [3, 1])
@test_throws AssertionError s.check_sortedindices(5, [0, 2])
@test s.check_sortedindices(5, Int[]) == nothing
@test s.check_sortedindices(5, [1, 3]) == nothing

@test s.check_embed_indices([1,[3,5],10,2,[70,11]]) == true
@test s.check_embed_indices([1,3,1]) == false
@test s.check_embed_indices([1,[10,11],7,[3,1]]) == false
@test s.check_embed_indices([[10,3],5,6,[3,7]]) == false
@test s.check_embed_indices([]) == true
end # testset
