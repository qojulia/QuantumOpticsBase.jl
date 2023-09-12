using Test
using QuantumOpticsBase
using LinearAlgebra, SparseArrays, Random

mutable struct test_operators{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}
  basis_l::BL
  basis_r::BR
  data::Matrix{ComplexF64}
  test_operators(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new{typeof(b1),typeof(b2)}(b1, b2, data) : throw(DimensionMismatch())
end

@testset "operators" begin

Random.seed!(0)

b1 = GenericBasis(3)
b2 = GenericBasis(2)
b = b1 ⊗ b2
op1 = randoperator(b1)
op2 = randoperator(b2)
op = randoperator(b, b)
op_test = test_operators(b, b, op.data)
op_test2 = test_operators(b1, b, randoperator(b1, b).data)
op_test3 = test_operators(b1 ⊗ b2, b2 ⊗ b1, randoperator(b, b).data)
ψ = randstate(b)
ρ = randoperator(b)

@test basis(op1) == b1
@test length(op1) == length(op1.data) == length(b1)^2

@test isequal(+op_test, op_test)

@test_throws ArgumentError op_test*op_test
@test_throws ArgumentError -op_test

@test_throws ArgumentError 1 + op_test
@test_throws ArgumentError op_test + 1
@test_throws ArgumentError 1 - op_test
@test_throws ArgumentError op_test - 1

@test_throws ArgumentError dagger(op_test)
@test_throws ArgumentError op_test'
@test_throws ArgumentError identityoperator(test_operators, b, b)
@test_throws ArgumentError tr(op_test)
@test_throws ArgumentError ptrace(op_test, [1])
@test_throws ArgumentError ishermitian(op_test)
@test_throws MethodError dense(op_test)
@test_throws ArgumentError sparse(op_test)
@test_throws ArgumentError transpose(op_test)

@test expect(1, op1, ρ) ≈ expect(embed(b, 1, op1), ρ)
@test expect(1, op1, ψ) ≈ expect(embed(b, 1, op1), ψ)
@test expect(op, [ρ, ρ]) == [expect(op, ρ) for i=1:2]
@test expect(1, op1, [ρ, ψ]) == [expect(1, op1, ρ), expect(1, op1, ψ)]

@test variance(1, op1, ρ) ≈ variance(embed(b, 1, op1), ρ)
@test variance(1, op1, ψ) ≈ variance(embed(b, 1, op1), ψ)
@test variance(op, [ρ, ρ]) == [variance(op, ρ) for i=1:2]
@test variance(1, op1, [ρ, ψ]) == [variance(1, op1, ρ), variance(1, op1, ψ)]

@test tensor(op_test) === op_test
@test_throws ArgumentError tensor(op_test, op_test)
@test_throws ArgumentError permutesystems(op_test, [1, 2])

@test embed(b, b, [1,2], op) == embed(b, [1,2], op)
@test embed(b, Dict{Vector{Int}, SparseOpType}()) == identityoperator(b)
@test_throws QuantumOpticsBase.IncompatibleBases embed(b1⊗b2, [2], [op1])

b_comp = b⊗b
@test isapprox(embed(b_comp, [1,[3,4]], [op1,op]), dense(op1 ⊗ one(b2) ⊗ op))
@test isapprox(embed(b_comp, [[1,2],4], [op,op2]), dense(op ⊗ one(b1) ⊗ op2))
@test_throws QuantumOpticsBase.IncompatibleBases embed(b_comp, [[1,2],3], [op,op2])
@test_throws QuantumOpticsBase.IncompatibleBases embed(b_comp, [[1,3],4], [op,op2])

function basis_vec(n, N)
    x = zeros(Complex{Float64}, N)
    x[n+1] = 1
    return x
end
function basis_maker(dims...)
    function bm(ns...)
        bases = [basis_vec(n, dim) for (n, dim) in zip(ns, dims)][end:-1:1]
        return reduce(kron, bases)
    end
end

embed_op = embed(b_comp, [1,4], op)
bv = basis_maker(3,2,3,2)
all_idxs = [(idx, jdx) for (idx, jdx) in [Iterators.product(0:1, 0:2)...]]

m11 = reshape([Bra(b_comp, bv(0,idx,jdx,0)) * embed_op * Ket(b_comp, bv(0,kdx,ldx,0))
              for ((idx, jdx), (kdx, ldx)) in Iterators.product(all_idxs, all_idxs)], (6,6))
@test isapprox(m11 / op.data[1, 1], diagm(0=>ones(Complex{Float64}, 6)))

m21 = reshape([Bra(b_comp, bv(1,idx,jdx,0)) * embed_op * Ket(b_comp, bv(0,kdx,ldx,0))
              for ((idx, jdx), (kdx, ldx)) in Iterators.product(all_idxs, all_idxs)], (6,6))
@test isapprox(m21 / op.data[2,1], diagm(0=>ones(Complex{Float64}, 6)))

m12 = reshape([Bra(b_comp, bv(0,idx,jdx,0)) * embed_op * Ket(b_comp, bv(1,kdx,ldx,0))
              for ((idx, jdx), (kdx, ldx)) in Iterators.product(all_idxs, all_idxs)], (6,6))
@test isapprox(m12 / op.data[1,2], diagm(0=>ones(Complex{Float64}, 6)))


b_comp = b_comp⊗b_comp
OP_test1 = dense(tensor([op1,one(b2),op,one(b1),one(b2),op1,one(b2)]...))
OP_test2 = embed(b_comp, [1,[3,4],7], [op1,op,op1])
@test isapprox(OP_test1, OP_test2)

b8 = b2⊗b2⊗b2
cnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
op_cnot = DenseOperator(b2⊗b2, cnot)
OP_cnot = embed(b8, [1,3], op_cnot)
@test ptrace(OP_cnot, [2])/2. == op_cnot
@test reduced(OP_cnot, [1,3])/2. == op_cnot
@test_throws AssertionError embed(b2⊗b2, [1,1], op_cnot)

b = FockBasis(40)
alpha = 1+5im
H = alpha * create(b) - conj(alpha) * destroy(b)
@test exp(sparse(H); threshold=1e-10) ≈ displace(b, alpha)
@test exp(sparse(zero(identityoperator(b)))) ≈ identityoperator(b)

@test one(b1).data == Diagonal(ones(b1.shape[1]))
@test one(op1).data == Diagonal(ones(b1.shape[1]))

@test_throws ArgumentError conj!(op_test)

bfock = FockBasis(2)
bspin = SpinBasis(1//2)
b = bfock⊗bspin⊗bspin

op12 = destroy(bfock)⊗sigmap(bspin)

@test embed(b, [1,2], op12) == destroy(bfock)⊗sigmap(bspin)⊗one(bspin)
@test embed(b, [1,3], op12) == destroy(bfock)⊗one(bspin)⊗sigmap(bspin)

# size of AbstractOperator
b1, b2 = NLevelBasis.((2, 3))
Lop1 = LazyTensor(b1^2, b2^2, 2, sparse(randoperator(b1, b2)))
@test size(Lop1) == size(dense(Lop1)) == size(dense(Lop1).data)
@test all(size(Lop1, k) == size(dense(Lop1), k) for k=1:4)
@test_throws ErrorException size(Lop1,  0)
@test_throws ErrorException size(Lop1, -1)
@test_throws ErrorException size(dense(Lop1),  0) # check for consistency
@test_throws ErrorException size(dense(Lop1), -1)

# issue 106 | https://github.com/qojulia/QuantumOpticsBase.jl/issues/106
a = destroy(FockBasis(5))
@test dagger(a^2) == dagger(a)^2
@test convert(Base._return_type(*, Tuple{typeof(a'), typeof(a')}), a') == a'

end # testset
