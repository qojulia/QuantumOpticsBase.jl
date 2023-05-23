using Test
using QuantumOpticsBase
using LinearAlgebra, Random

@testset "time-dependent operators" begin
    # TODO:
    #  * Test time mismatch errors
    #  * Test set_time!() and other new interfaces

    o = TimeDependentSum(ComplexF64, FockBasis(2), FockBasis(3))
    @test length(suboperators(o)) == 0
    psi = randstate(FockBasis(3))
    @test norm(o(1.1) * psi) == 0.0
    @test basis(o(1.1) * psi) == FockBasis(2)
    @test o + o == o
    @test (o * o')(0.0) == TimeDependentSum(ComplexF64, FockBasis(2), FockBasis(2))
    @test (o')(0.0) == TimeDependentSum(ComplexF64, FockBasis(3), FockBasis(2))
    @test 2o == o
    @test dense(o) == o
    @test sparse(o) == o

    o = TimeDependentSum(ComplexF64, FockBasis(2), FockBasis(2))
    b = FockBasis(2) ⊗ FockBasis(3)

    n = number(FockBasis(2))
    a = randoperator(FockBasis(2))

    f1, f2 = 1.0im, t->t*3.0
    t = randn()

    o_t1 = TimeDependentSum(f1, a)
    @test eval_coefficients(o_t1, t) == [f1]
    @test o + o_t1 == o_t1
    @test o - o_t1 == -o_t1
    @test iszero(dense(static_operator(o * o_t1)))

    #o_t1_comp = embed_lazy(FockBasis(2) ⊗ GenericBasis(2), 1, o_t1)
    #@test eval_coefficients(o_t1_comp, t) == [f1]
    #@test isa(suboperators(o_t1_comp)[1], LazyTensor)

    o_t1_comp2 = embed(FockBasis(2) ⊗ GenericBasis(2), 1, o_t1)
    @test eval_coefficients(o_t1_comp2, t) == [f1]
    @test isa(suboperators(o_t1_comp2)[1], SparseOpType)
    
    o_t = TimeDependentSum([f1, f2], [a,n])

    @test (@inferred eval_coefficients(o_t, t)) == [1.0im, t*3.0]

    # function-call interface
    @test static_operator(o_t(t)).factors == [1.0im, t*3.0 + 0.0im]
    @test all(static_operator(o_t(t)).operators .=== (a, n))

    o_t_tup = TimeDependentSum(Tuple, o_t)
    @test static_operator(o_t_tup(t)).factors == [1.0im, t*3.0 + 0.0im]
    @test all(static_operator(o_t_tup(t)).operators .=== (a, n))
    @test (@allocated o_t_tup(t)) == 0

    o_t2 = TimeDependentSum(f1=>a, f2=>n)
    @test o_t(t) == o_t2(t)

    o_t_ = dense(o_t)
    @test all(isa(o, DenseOpType) for o in suboperators(o_t_))
    @test o_t_.coefficients == o_t.coefficients

    o_t_ = sparse(o_t)
    @test all(isa(o, SparseOpType) for o in suboperators(o_t_))
    @test o_t_.coefficients == o_t.coefficients

    o_t_ = dagger(o_t)
    @test all(isa(o.data, Adjoint) for o in suboperators(o_t_))
    @test eval_coefficients(o_t_, t) == conj.(eval_coefficients(o_t, t))

    b = randoperator(FockBasis(2)) #LazyProduct(a)  LazyProduct appears to break things??
    o2_t = TimeDependentSum([t->cos(t), t->t/3.0], [b, n])
    
    # operations
    o_res = o_t(0.0) + o2_t(0.0)
    @test isa(o_res, TimeDependentSum)
    @test eval_coefficients(o_res, t) == ComplexF64[1.0im, t*3.0, cos(t), t/3.0]
    @test all(suboperators(o_res) .=== (a, n, b, n))

    o_res = o_t - o2_t
    @test isa(o_res, TimeDependentSum)
    @test eval_coefficients(o_res, t) == ComplexF64[1.0im, t*3.0, -cos(t), -t/3.0]
    @test all(suboperators(o_res) .=== (a, n, b, n))

    o_res = -o_t
    @test isa(o_res, TimeDependentSum)
    @test eval_coefficients(o_res, t) == [-1.0im, -t*3.0]
    @test all(suboperators(o_res) .=== (a, n))
    
    o_res = a + o2_t
    @test isa(o_res, TimeDependentSum)
    @test eval_coefficients(o_res, t) == ComplexF64[1.0, cos(t), t/3.0]
    @test all(suboperators(o_res) .=== (a, b, n))

    o_res = a - o2_t
    @test isa(o_res, TimeDependentSum)
    @test eval_coefficients(o_res, t) == ComplexF64[1.0, -cos(t), -t/3.0]
    @test all(suboperators(o_res) .=== (a, b, n))
    
    o_res = LazySum(a, n) + o2_t
    @test isa(o_res, TimeDependentSum)
    @test eval_coefficients(o_res, t) == ComplexF64[1.0, 1.0, cos(t), t/3.0]
    @test all(suboperators(o_res) .=== (a, n, b, n))

    fac = randn(ComplexF64)
    o_res = fac * o2_t
    @test isa(o_res, TimeDependentSum)
    @test [eval_coefficients(o_res, t)...] ≈ [fac*cos(t), fac*t/3.0] rtol=1e-10
    @test all(suboperators(o_res) .=== (b, n))

    o_res = o_t * o2_t
    @test isa(o_res, TimeDependentSum)
    @test [eval_coefficients(o_res, t)...] ≈ [(c1*c2 for (c1,c2) in Iterators.product(eval_coefficients(o_t, t), eval_coefficients(o2_t, t)))...] rtol=1e-10
    @test dense(static_operator(o_res(t))).data ≈ (dense(static_operator(o_t(t))) * dense(static_operator(o2_t(t)))).data rtol=1e-10

    V = identityoperator(basis(o_t), GenericBasis(length(basis(o_t))))
    o_res = o_t * V
    @test isa(o_res, TimeDependentSum)
    @test [eval_coefficients(o_res, t)...] ≈ [eval_coefficients(o_t, t)...] rtol=1e-10
    @test dense(static_operator(o_res(t))).data ≈ dense(static_operator(o_t(t))).data rtol=1e-10
end

@testset "time-dependent operators: Shifts, restrictions, and stretchs" begin
    b = FockBasis(1)
    op = TimeDependentSum((1.0, identity), (number(b), destroy(b)))
    op_shift = timeshift(op, 10.0)
    @test @inferred eval_coefficients(op_shift, 0.0) == (1.0, -10.0)
    @test @inferred eval_coefficients(op_shift, 10.0) == (1.0, 0.0)
    @test @inferred eval_coefficients(op_shift, 20.0) == (1.0, 10.0)

    op_block_shift = timerestrict(op_shift, 5.0, 15.0)
    @test @inferred eval_coefficients(op_block_shift, 4.9) == (0.0, 0.0)
    @test @inferred eval_coefficients(op_block_shift, 5.0) == (1.0, -5.0)
    @test @inferred eval_coefficients(op_block_shift, 20.0) == (0.0, 0.0)
    
    op_stretch_block_shift = timestretch(op_block_shift, 2.0)
    @test @inferred eval_coefficients(op_stretch_block_shift, 9.9) == (0.0, 0.0)
    @test @inferred eval_coefficients(op_stretch_block_shift, 10.0) == (1.0, -5.0)
    @test @inferred eval_coefficients(op_stretch_block_shift, 30.0-√eps()) == (1.0, 5.0-0.5√eps())
    @test @inferred eval_coefficients(op_stretch_block_shift, 30.0) == (0.0, 0.0)
end
