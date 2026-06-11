# test/test_sciml_lazy_operators.jl
# Correctness tests for the SciMLOperators package extension.
# Discovered automatically by TestItemRunner — no runtests.jl edit needed.

@testitem "SciMLOperators: wrapper type and exports" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    op = sigmax(b0) ⊗ sigmaz(b0)
    w  = sciml_lazy_operator(op)
    @test w isa AbstractOperator
    @test hasproperty(w, :sciml_op)
    @test w.basis_l == op.basis_l
    @test w.basis_r == op.basis_r
end

@testitem "SciMLOperators: dense Operator" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    op = sigmax(b0) ⊗ sigmaz(b0)
    @test dense(sciml_lazy_operator(op)) ≈ dense(op)
end

@testitem "SciMLOperators: sparse Operator" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    op = SparseOperator(sigmax(b0) ⊗ sigmaz(b0))
    @test dense(sciml_lazy_operator(op)) ≈ dense(op)
end

@testitem "SciMLOperators: LazySum uniform factors" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    H = LazySum(LazyTensor(b4,1,sx), LazyTensor(b4,2,sz), LazyTensor(b4,3,sx))
    w = sciml_lazy_operator(H)
    @test dense(w) ≈ dense(H)
    @test w * psi  ≈ H * psi
end

@testitem "SciMLOperators: LazySum non-unity factors" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    H = LazySum([0.5+0im, -1.2im], (LazyTensor(b4,1,sx), LazyTensor(b4,3,sz)))
    w = sciml_lazy_operator(H)
    @test dense(w) ≈ dense(H)
    @test w * psi  ≈ H * psi
end

@testitem "SciMLOperators: LazyProduct" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    P = LazyProduct(LazyTensor(b4,1,sx), LazyTensor(b4,2,sz))
    w = sciml_lazy_operator(P)
    @test dense(w) ≈ dense(P)
    @test w * psi  ≈ P * psi
end

@testitem "SciMLOperators: LazyProduct scalar factor" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    P = 3.0 * LazyProduct(LazyTensor(b4,1,sx), LazyTensor(b4,2,sz))
    w = sciml_lazy_operator(P)
    @test dense(w) ≈ dense(P)
    @test w * psi  ≈ P * psi
end

@testitem "SciMLOperators: LazyTensor edge site dense" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, 1, sigmax(b0))
    w  = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: LazyTensor edge site sparse" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, 4, SparseOperator(sigmaz(b0)))
    w  = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: LazyTensor mid site dense" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, 2, sigmay(b0))
    w  = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: LazyTensor mid site sparse" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, 3, SparseOperator(sigmax(b0)))
    w  = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: LazyTensor multi-site" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, [1, 3], (sx, sz))
    w  = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: LazyTensor scalar factor" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, 2, sigmax(b0), 2.5)
    w  = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: LazyTensor spin-1 local dim" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b1   = SpinBasis(1)
    b4_1 = tensor(b1,b1,b1,b1)
    psi  = let v = randn(ComplexF64, length(b4_1)); Ket(b4_1, v ./ norm(v)) end
    lt   = LazyTensor(b4_1, 2, sigmax(b1))
    w    = sciml_lazy_operator(lt)
    @test dense(w) ≈ dense(lt)
    @test w * psi  ≈ lt * psi
end

@testitem "SciMLOperators: mixed TFIM Hamiltonian" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    H = LazySum(
        LazyTensor(b4,[1,2],(sz,sz)), LazyTensor(b4,[2,3],(sz,sz)),
        LazyTensor(b4,[3,4],(sz,sz)),
        0.5*LazyTensor(b4,1,sx), 0.5*LazyTensor(b4,2,sx),
        0.5*LazyTensor(b4,3,sx), 0.5*LazyTensor(b4,4,sx),
    )
    w = sciml_lazy_operator(H)
    @test dense(w) ≈ dense(H) atol=1e-10
    @test w * psi  ≈ H * psi  atol=1e-10
end

@testitem "SciMLOperators: exact example from issue 522" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b  = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    H = LazySum(
        LazyTensor(b,1,sx),
        LazyTensor(b,2,sz),
        LazyProduct(LazyTensor(b,3,sx), LazyTensor(b,4,sz)),
    )
    H_sciml = sciml_lazy_operator(H)
    psi = Ket(b, randn(ComplexF64, length(b)))
    @test dense(H_sciml) ≈ dense(H)
    @test H_sciml * psi  ≈ H * psi
end

@testitem "SciMLOperators: cache_sciml_lazy_operator" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    H  = LazySum(LazyTensor(b4,1,sx), LazyTensor(b4,3,sz))
    w  = sciml_lazy_operator(H)
    wc = cache_sciml_lazy_operator(w, psi.data)
    @test wc isa SciMLOperatorWrapper
    @test wc * psi ≈ w * psi
end

@testitem "SciMLOperators: mul! 5-arg form" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    psi = let v = randn(ComplexF64, length(b4)); Ket(b4, v ./ norm(v)) end
    lt = LazyTensor(b4, 2, sigmaz(b0))
    w  = sciml_lazy_operator(lt)
    α, β = 1.5+0im, 0.5+0im
    r_sciml = Ket(b4, randn(ComplexF64, length(b4)))
    r_ref   = copy(r_sciml)
    mul!(r_sciml, w,  psi, α, β)
    mul!(r_ref,   lt, psi, α, β)
    @test r_sciml ≈ r_ref
end

@testitem "SciMLOperators: scalar arithmetic" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    w = sciml_lazy_operator(LazyTensor(b4, 1, sigmax(b0)))
    @test dense(2.0 * w) ≈ 2.0 * dense(w)
    @test dense(w * 3.0) ≈ 3.0 * dense(w)
    @test dense(w / 4.0) ≈ (1/4.0) * dense(w)
end

@testitem "SciMLOperators: operator addition" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    wa = sciml_lazy_operator(LazyTensor(b4,1,sx))
    wb = sciml_lazy_operator(LazyTensor(b4,3,sz))
    @test dense(wa + wb) ≈ dense(LazyTensor(b4,1,sx)) + dense(LazyTensor(b4,3,sz))
end

@testitem "SciMLOperators: operator composition" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    sx, sz = sigmax(b0), sigmaz(b0)
    wa = sciml_lazy_operator(LazyTensor(b4,1,sx))
    wb = sciml_lazy_operator(LazyTensor(b4,1,sz))
    @test dense(wa * wb) ≈ dense(LazyTensor(b4,1,sx)) * dense(LazyTensor(b4,1,sz))
end

@testitem "SciMLOperators: dagger" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    H = LazySum(LazyTensor(b4,1,sigmap(b0)), LazyTensor(b4,3,sigmam(b0)))
    w = sciml_lazy_operator(H)
    @test dense(dagger(w)) ≈ dagger(dense(H))
end

@testitem "SciMLOperators: basis preserved through arithmetic" tags=[:sciml] begin
    using QuantumOpticsBase, SciMLOperators, LinearAlgebra
    b0 = SpinBasis(1//2)
    b4 = tensor(b0,b0,b0,b0)
    w = sciml_lazy_operator(LazyTensor(b4, 2, sigmax(b0)))
    @test (2.0 * w).basis_l == b4
    @test dagger(w).basis_l == b4
    @test dagger(w).basis_r == b4
end
