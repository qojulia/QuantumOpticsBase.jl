using Random
using Test
using SciMLOperators
using QuantumOpticsBase

Test.@testset "test_sciml_lazyoperators" begin
    rng = MersenneTwister(42)
    b = SpinBasis(1//2)
    chain = b ⊗ b ⊗ b ⊗ b

    sx = sigmax(b)
    sz = sigmaz(b)

    H_current = LazySum(
        LazyTensor(chain, chain, [1], (sx,)),
        LazyTensor(chain, chain, [2], (sz,)),
        LazyProduct(
            LazyTensor(chain, chain, [3], (sx,)),
            LazyTensor(chain, chain, [4], (sz,)),
        ),
    )

    H_sciml = QuantumOpticsBase.sciml_lazy_operator(H_current)
    H_dense = dense(H_current)
    ψ = Ket(chain, randn(rng, ComplexF64, length(chain)))

    Test.@test H_sciml.basis_l == H_current.basis_l
    Test.@test H_sciml.basis_r == H_current.basis_r
    Test.@test dense(H_sciml).data ≈ H_dense.data
    Test.@test (H_sciml * ψ).data ≈ (H_current * ψ).data
    Test.@test (2.0 * H_sciml * ψ).data ≈ ((2.0 * H_current) * ψ).data
    Test.@test (dagger(H_sciml) * ψ).data ≈ (dagger(H_current) * ψ).data
    Test.@test dense(H_sciml + H_sciml).data ≈ dense(2 * H_current).data

    A = LazyTensor(chain, chain, [1], (sx,))
    B = LazyTensor(chain, chain, [2], (sz,))
    A_sciml = QuantumOpticsBase.sciml_lazy_operator(A)
    B_sciml = QuantumOpticsBase.sciml_lazy_operator(B)
    Test.@test dense(A_sciml * B_sciml).data ≈ dense(A * B).data

    local_dense = DenseOperator(chain, chain, randn(rng, ComplexF64, length(chain), length(chain)))
    local_sparse = SparseOperator(chain, sparse(local_dense.data))

    Test.@test dense(QuantumOpticsBase.sciml_lazy_operator(local_dense)).data ≈ local_dense.data
    Test.@test dense(QuantumOpticsBase.sciml_lazy_operator(local_sparse)).data ≈ local_sparse.data

    cached = cache_operator(H_sciml, ψ.data)
    Test.@test (cached * ψ).data ≈ (H_sciml * ψ).data
end
