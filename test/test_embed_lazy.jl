@testitem "test_embed_lazy" begin
using Test
using QuantumOpticsBase
using LinearAlgebra, SparseArrays, Random

@testset "embed_lazy" begin

Random.seed!(0)

b0 = SpinBasis(1//2)
b = tensor(b0, b0, b0, b0)

sx = sigmax(b0)
sz = sigmaz(b0)

psi = Ket(b, randn(ComplexF64, length(b)))

@testset "single-site DataOperator" begin
    op_eager = embed(b, 2, sx)
    op_lazy = embed_lazy(b, 2, sx)

    @test op_lazy isa LazyTensor
    @test dense(op_lazy) == dense(op_eager)
    @test op_lazy * psi == op_eager * psi

    @test embed_lazy(b, 2, dense(sx)) isa LazyTensor
    @test embed_lazy(b, 2, sparse(sx)) isa LazyTensor
    @test dense(embed_lazy(b, 2, sparse(sx))) == dense(embed(b, 2, sparse(sx)))
    @test dense(embed_lazy(b, [2], sx)) == dense(embed(b, 2, sx))
end

@testset "LazySum" begin
    local_h = LazySum(sx, 0.5 * sz)
    lazy_h = embed_lazy(b, 3, local_h)
    eager_h = embed(b, 3, local_h)

    @test lazy_h isa LazySum
    @test all(t -> t isa LazyTensor, lazy_h.operators)
    @test lazy_h.factors == local_h.factors
    @test dense(lazy_h) == dense(eager_h)
    @test lazy_h * psi == eager_h * psi

    H = sum(embed_lazy(b, i, local_h) for i in 1:4)
    H_eager = sum(embed(b, i, local_h) for i in 1:4)
    @test H isa LazySum
    @test dense(H) ≈ dense(H_eager)
end

@testset "multi-site vector of operators" begin
    op_lazy12 = embed_lazy(b, [1, 3], [sx, sz])
    op_eager12 = embed(b, [1, 3], [sx, sz])
    @test op_lazy12 isa LazyTensor
    @test dense(op_lazy12) == dense(op_eager12)
    @test op_lazy12 * psi == op_eager12 * psi
end

@testset "asymmetric bases" begin
    bf = FockBasis(2)
    b_l = b0 ⊗ bf
    b_r = bf ⊗ b0
    a = randoperator(bf, b0)
    op_asym = embed_lazy(b_l, b_r, 2, a)
    @test op_asym isa LazyTensor
    @test dense(op_asym) == dense(embed(b_l, b_r, 2, a))
end

@testset "LazyTensor re-embedding" begin
    b1a = GenericBasis(2)
    b2a = GenericBasis(1)
    b3a = GenericBasis(6)
    b1b = GenericBasis(3)
    b2b = GenericBasis(4)
    b3b = GenericBasis(5)
    b_l = b1a ⊗ b2a ⊗ b3a
    b_r = b1b ⊗ b2b ⊗ b3b
    op1 = randoperator(b1a, b1b)
    op3 = randoperator(b3a, b3b)
    x = LazyTensor(b_l, b_r, [1, 3], (op1, sparse(op3)), 0.3)
    x_sub = LazyTensor(b1a ⊗ b3a, b1b ⊗ b3b, [1, 2], (op1, sparse(op3)), 0.3)
    @test embed_lazy(b_l, b_r, [1, 3], x_sub) == x

    bsub = b0 ⊗ b0
    lt_partial = LazyTensor(bsub, [2], (sz,), 2.0)
    lz = embed_lazy(b, [1, 3], lt_partial)
    @test lz isa LazyTensor
    @test lz.indices == [3]
    @test lz.factor == 2.0
    @test dense(lz) ≈ dense(2.0 * embed(b, 3, sz))
end

@testset "TimeDependentSum" begin
    subop = randoperator(b0)
    td_op = TimeDependentSum(ComplexF64, (t -> cos(t)) => subop)
    td_lazy = embed_lazy(b, 2, td_op)
    @test td_lazy isa TimeDependentSum
    @test dense(td_lazy) == dense(embed(b, 2, td_op))
    @test all(t -> t isa LazyTensor, static_operator(td_lazy).operators)
end

@testset "errors" begin
    @test_throws QuantumOpticsBase.IncompatibleBases embed_lazy(b, 1, destroy(FockBasis(2)))
    @test_throws ArgumentError embed_lazy(b, 1, LazyProduct(sx))
    @test_throws ArgumentError embed_lazy(b, [1, 2], sx ⊗ sz)
    @test_throws ArgumentError embed_lazy(b, [3, 1], [sz, sx])
    @test_throws ArgumentError embed_lazy(b, [3, 1], LazyTensor(b0 ⊗ b0, [1, 2], (sx, sz)))
end

@testset "allocation" begin
    b_chain = reduce(⊗, ntuple(_ -> SpinBasis(1//2), 12))
    alloc_lazy = @allocated embed_lazy(b_chain, 4, sx)
    alloc_eager = @allocated embed(b_chain, 4, sx)
    @test alloc_lazy < 10_000
    @test alloc_lazy < alloc_eager ÷ 10
end

end # testset
end
