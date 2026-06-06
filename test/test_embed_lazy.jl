@testitem "test_embed_lazy" begin
using Test
using QuantumOpticsBase
using Random, SparseArrays, LinearAlgebra

@testset "embed_lazy" begin

Random.seed!(0)

b1 = NLevelBasis(3)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)
b = b1 ⊗ b2 ⊗ b3

I2 = dense(identityoperator(b2))

op1 = DenseOperator(b1, b1, rand(ComplexF64, length(b1), length(b1)))
op2 = DenseOperator(b2, b2, rand(ComplexF64, length(b2), length(b2)))
op3 = DenseOperator(b3, b3, rand(ComplexF64, length(b3), length(b3)))

psi = Ket(b, rand(ComplexF64, length(b)))

# single-index dense operator
lz = embed_lazy(b, 1, op1)
@test lz isa LazyTensor
@test dense(lz) ≈ dense(embed(b, 1, op1))
@test (lz * psi).data ≈ (embed(b, 1, op1) * psi).data

lz = embed_lazy(b, 2, op2)
@test lz isa LazyTensor
@test dense(lz) ≈ dense(embed(b, 2, op2))
@test (lz * psi).data ≈ (embed(b, 2, op2) * psi).data

@test dense(embed_lazy(b, [3], op3)) ≈ dense(embed(b, 3, op3))

# single-index sparse operator
lz = embed_lazy(b, 1, sparse(op1))
@test lz isa LazyTensor
@test dense(lz) ≈ dense(embed(b, 1, op1))
@test (lz * psi).data ≈ (embed(b, 1, op1) * psi).data

@test dense(embed_lazy(b, b, 2, op2)) ≈ dense(embed(b, 2, op2))

# LazySum stays a LazySum of lazily embedded terms, factors preserved
local_h = LazySum([1.0, 0.5], (op2, sparse(op2)))
lz = embed_lazy(b, 2, local_h)
@test lz isa LazySum
@test all(t -> t isa LazyTensor, lz.operators)
@test dense(lz) ≈ dense(embed(b, 2, local_h))
@test (lz * psi).data ≈ (embed(b, 2, local_h) * psi).data
@test collect(lz.factors) == collect(local_h.factors)

# re-embedding a LazyTensor across several subsystems keeps its suboperators and factor
bsub = b1 ⊗ b3
lt = LazyTensor(bsub, [1, 2], (op1, op3), 2.0)
lz = embed_lazy(b, [1, 3], lt)
@test lz isa LazyTensor
@test lz.factor == lt.factor
@test lz.indices == [1, 3]
@test dense(lz) ≈ 2.0 * dense(op1 ⊗ I2 ⊗ op3)
@test (lz * psi).data ≈ (2.0 * (op1 ⊗ I2 ⊗ op3) * psi).data

# a LazyTensor occupying only some of the addressed sites maps indices correctly
lt_partial = LazyTensor(bsub, [2], (op3,), 1.0)
lz = embed_lazy(b, [1, 3], lt_partial)
@test lz.indices == [3]
@test dense(lz) ≈ dense(embed(b, 3, op3))

lt_sp = LazyTensor(bsub, [1, 2], (sparse(op1), sparse(op3)), 1.0)
@test dense(embed_lazy(b, [1, 3], lt_sp)) ≈ dense(op1 ⊗ I2 ⊗ op3)

# TimeDependentSum keeps its coefficients and embeds the static terms lazily
f = t -> 2.0 + 0.0im
tds = TimeDependentSum([f], [op2])
lz = embed_lazy(b, 2, tds)
@test lz isa TimeDependentSum
set_time!(lz, 1.3)
eager = embed(b, 2, tds); set_time!(eager, 1.3)
@test dense(static_operator(lz)) ≈ dense(static_operator(eager))
@test static_operator(lz) isa LazySum
@test all(t -> t isa LazyTensor, static_operator(lz).operators)

@test_throws QuantumOpticsBase.IncompatibleBases embed_lazy(b, 1, op2)
@test_throws QuantumOpticsBase.IncompatibleBases embed_lazy(b, [1, 3], LazyTensor(b1 ⊗ b2, [1, 2], (op1, op2), 1.0))

@test_throws ArgumentError embed_lazy(b, [1, 3], op1 ⊗ op3)
@test_throws ArgumentError embed_lazy(b, [3, 1], lt)

# the lazy path must not materialize the full embedded matrix
big = reduce(⊗, fill(b2, 10))
sx = sigmax(b2)
embed_lazy(big, 1, sx); embed(big, 1, sx)
alloc_lazy = @allocated embed_lazy(big, 1, sx)
alloc_eager = @allocated embed(big, 1, sx)
@test alloc_lazy < alloc_eager

end # testset
end
