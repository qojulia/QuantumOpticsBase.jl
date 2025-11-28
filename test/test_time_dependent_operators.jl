@testitem "test_time_dependent_operators" begin
using Test
using QuantumOpticsBase
using LinearAlgebra, Random

function api_test(op)
    for func in (basis, length, size, tr, normalize, normalize!, identityoperator, one, eltype)
        func(op)
    end
    for func in (ptrace,)
        func(op, [1])
    end
end

@testset "time-dependent operators" begin
    QOB = QuantumOpticsBase

    subop = randoperator(FockBasis(1))
    op = LazyTensor(FockBasis(1) ⊗ FockBasis(3), 1, subop)
    @test QOB.static_operator(op) === op
    @test set_time!(op, 1.0) === op
    @test_throws ArgumentError current_time(op)
    @test QOB.is_const(op)

    op = dense(op)

    o = TimeDependentSum((t->2.0*cos(t))=>op)
    @test !QOB.is_const(o)
    @test TimeDependentSum(o) === o
    subo = TimeDependentSum(2.0=>subop)
    @test QOB.suboperators(o)[1] == op
    @test QOB.suboperators(QOB.static_operator(o))[1] == op

    api_test(op)
    api_test(o)
    api_test(subo)

    psi = randstate(basis(o))
    for f in (expect, variance)
        for s in (psi, op)
            @test f(o, s) ≈ f(2op, s)
            @test f(1, subo, s) ≈ f(1, 2subop, s)
            @test f([1], subo, s) ≈ f([1], 2subop, s)
        end
    end

    set_time!(o, 0.5)
    o_copy = copy(o)
    @test o_copy == o
    @test isequal(o_copy, o)

    set_time!(o, 1.0)
    @test current_time(o) == 1.0
    @test current_time(o_copy) == 0.5
    @test o_copy != o
    @test !isequal(o_copy, o)
    @test QOB.static_operator(o) != QOB.static_operator(o_copy)

    ls = LazySum(o)
    @test !QOB.is_const(ls)
    set_time!(ls, 0.0)
    @test current_time(o) == 0.0

    api_test(ls)

    tdls = TimeDependentSum([1.0], ls; init_time=1.0)
    @test current_time(o) == 1.0

    api_test(tdls)

    o = TimeDependentSum(ComplexF64, FockBasis(2), FockBasis(3))
    @test length(QOB.suboperators(o)) == 0
    psi = randstate(FockBasis(3))
    @test norm(o(1.1) * psi) == 0.0
    @test basis(o(1.1) * psi) == FockBasis(2)
    @test o + o == o
    @test (o * o')(0.0) == TimeDependentSum(ComplexF64, FockBasis(2), FockBasis(2))
    @test (o')(0.0) == TimeDependentSum(ComplexF64, FockBasis(3), FockBasis(2))
    @test 2o == o
    @test dense(o) == o
    @test sparse(o) == o

    @test o(0.0) != o(1.0)

    o = TimeDependentSum(ComplexF64, FockBasis(2), FockBasis(2))
    b = FockBasis(2) ⊗ FockBasis(3)

    n = number(FockBasis(2))
    a = randoperator(FockBasis(2))

    f1, f2 = 1.0im, t->t*3.0
    t = randn()

    o_t1 = TimeDependentSum(f1, a)
    @test QOB.is_const(o_t1)
    @test QOB.eval_coefficients(o_t1, t) == [f1]
    @test o + o_t1 == o_t1
    @test o - o_t1 == -o_t1
    @test iszero(dense(QOB.static_operator(o * o_t1)))

    #o_t1_comp = embed_lazy(FockBasis(2) ⊗ GenericBasis(2), 1, o_t1)
    #@test QOB.eval_coefficients(o_t1_comp, t) == [f1]
    #@test isa(QOB.suboperators(o_t1_comp)[1], LazyTensor)

    o_t1_comp2 = embed(FockBasis(2) ⊗ GenericBasis(2), 1, o_t1)
    @test QOB.eval_coefficients(o_t1_comp2, t) == [f1]
    @test isa(QOB.suboperators(o_t1_comp2)[1], SparseOpType)

    o_t1_comp2_ = embed(FockBasis(2) ⊗ GenericBasis(2), [1], o_t1)
    @test o_t1_comp2 == o_t1_comp2_

    o_t = TimeDependentSum([f1, f2], [a,n])
    @test !QOB.is_const(o_t)

    @test (@inferred QOB.eval_coefficients(o_t, t)) == [1.0im, t*3.0]

    # function-call interface
    @test QOB.static_operator(o_t(t)).factors == [1.0im, t*3.0 + 0.0im]
    @test all(QOB.static_operator(o_t(t)).operators .== (a, n))

    o_t_tup = TimeDependentSum(Tuple, o_t)
    @test QOB.static_operator(o_t_tup(t)).factors == [1.0im, t*3.0 + 0.0im]
    @test all(QOB.static_operator(o_t_tup(t)).operators .== (a, n))
    if VERSION.minor == 11 # issue #178 https://github.com/qojulia/QuantumOpticsBase.jl/issues/178
        @test_broken (@allocated set_time!(o_t_tup, t)) == 0
    else
        @test (@allocated set_time!(o_t_tup, t)) == 0
    end

    o_t2 = TimeDependentSum(f1=>a, f2=>n)
    @test o_t(t) == o_t2(t)

    o_t_ = dense(o_t)
    @test all(isa(o, DenseOpType) for o in QOB.suboperators(o_t_))
    @test o_t_.coefficients == o_t.coefficients

    o_t_ = sparse(o_t)
    @test all(isa(o, SparseOpType) for o in QOB.suboperators(o_t_))
    @test o_t_.coefficients == o_t.coefficients

    o_t_ = dagger(o_t)
    @test all(isa(o.data, Adjoint) for o in QOB.suboperators(o_t_))
    @test QOB.eval_coefficients(o_t_, t) == conj.(QOB.eval_coefficients(o_t, t))

    b = randoperator(FockBasis(2))
    o2_t = TimeDependentSum([t->cos(t), t->t/3.0], [b, n])

    # operations
    o_res = o_t(0.0) + o2_t(0.0)
    @test isa(o_res, TimeDependentSum)
    @test QOB.eval_coefficients(o_res, t) == ComplexF64[1.0im, t*3.0, cos(t), t/3.0]
    @test all(QOB.suboperators(o_res) .== (a, n, b, n))

    o_res = o_t - o2_t
    @test isa(o_res, TimeDependentSum)
    @test QOB.eval_coefficients(o_res, t) == ComplexF64[1.0im, t*3.0, -cos(t), -t/3.0]
    @test all(QOB.suboperators(o_res) .== (a, n, b, n))

    o_res = -o_t
    @test isa(o_res, TimeDependentSum)
    @test QOB.eval_coefficients(o_res, t) == [-1.0im, -t*3.0]
    @test all(QOB.suboperators(o_res) .== (a, n))

    o_res = a + o2_t
    @test isa(o_res, TimeDependentSum)
    @test QOB.eval_coefficients(o_res, t) == ComplexF64[1.0, cos(t), t/3.0]
    @test all(QOB.suboperators(o_res) .== (a, b, n))

    o_res = a - o2_t
    @test isa(o_res, TimeDependentSum)
    @test QOB.eval_coefficients(o_res, t) == ComplexF64[1.0, -cos(t), -t/3.0]
    @test all(QOB.suboperators(o_res) .== (a, b, n))

    o_res = LazySum(a, n) + o2_t
    @test isa(o_res, TimeDependentSum)
    @test QOB.eval_coefficients(o_res, t) == ComplexF64[1.0, 1.0, cos(t), t/3.0]
    @test all(QOB.suboperators(o_res) .== (a, n, b, n))

    fac = randn(ComplexF64)
    o_res = fac * o2_t
    @test isa(o_res, TimeDependentSum)
    @test [QOB.eval_coefficients(o_res, t)...] ≈ [fac*cos(t), fac*t/3.0] rtol=1e-10
    @test all(QOB.suboperators(o_res) .== (b, n))

    fac = randn(ComplexF64)
    o_res = o2_t / fac
    @test isa(o_res, TimeDependentSum)
    @test [QOB.eval_coefficients(o_res, t)...] ≈ [cos(t)/fac, t/3.0/fac] rtol=1e-10
    @test all(QOB.suboperators(o_res) .== (b, n))

    o_res = o_t * o2_t
    @test isa(o_res, TimeDependentSum)
    @test [QOB.eval_coefficients(o_res, t)...] ≈ [(c1*c2 for (c1,c2) in Iterators.product(QOB.eval_coefficients(o_t, t), QOB.eval_coefficients(o2_t, t)))...] rtol=1e-10
    @test dense(QOB.static_operator(o_res(t))).data ≈ (dense(QOB.static_operator(o_t(t))) * dense(QOB.static_operator(o2_t(t)))).data rtol=1e-10

    V = identityoperator(basis(o_t), GenericBasis(length(basis(o_t))))
    o_res = o_t * V
    @test isa(o_res, TimeDependentSum)
    @test [QOB.eval_coefficients(o_res, t)...] ≈ [QOB.eval_coefficients(o_t, t)...] rtol=1e-10
    @test dense(QOB.static_operator(o_res(t))).data ≈ dense(QOB.static_operator(o_t(t))).data rtol=1e-10

    # combinations with other operator types
    ls = LazySum(ComplexF64, basis(o_t), basis(o_t))
    ro = randoperator(basis(o_t))
    for op in (+, -, *)
        @test dense(QOB.static_operator(op(o_t(0.0), ls))) == dense(op(QOB.static_operator(o_t(0.0)), ls))
        @test dense(QOB.static_operator(op(o_t(0.0), ro))) == dense(op(QOB.static_operator(o_t(0.0)), ro))
        @test dense(QOB.static_operator(op(ls, o_t(0.0)))) == dense(op(ls, QOB.static_operator(o_t(0.0))))
        @test dense(QOB.static_operator(op(ro, o_t(0.0)))) == dense(op(ro, QOB.static_operator(o_t(0.0))))
    end

    # mul!
    for input in (randoperator(basis(o_t)), randstate(basis(o_t)))
        out = mul!(copy(input), o_t, input)
        out2 = mul!(copy(input), QOB.static_operator(o_t), input)
        @test out == out2
    end
    for input in (randoperator(basis(o_t)), randstate(basis(o_t))')
        out = mul!(copy(input), input, o_t)
        out2 = mul!(copy(input), input, QOB.static_operator(o_t))
        @test out == out2
    end

    # test eval_coefficients independently
    coeffs = (1.0, cos, 2.0, 1.0im)
    coeffs_at_0 = (1.0, 1.0, 2.0, 1.0im)
    for i in 1:4
        @test @inferred QOB.eval_coefficients(ComplexF64, coeffs[1:i], 0.0) == ComplexF64.(coeffs_at_0[1:i])
    end
end

@testset "time-dependent operators: Shifts, restrictions, and stretchs" begin
    QOB = QuantumOpticsBase

    b = FockBasis(1)
    op = TimeDependentSum((1.0, identity), (number(b), destroy(b)))
    op_shift = time_shift(op, 10.0)
    @test @inferred QOB.eval_coefficients(op_shift, 0.0) == (1.0, -10.0)
    @test @inferred QOB.eval_coefficients(op_shift, 10.0) == (1.0, 0.0)
    @test @inferred QOB.eval_coefficients(op_shift, 20.0) == (1.0, 10.0)

    op_block_shift = time_restrict(op_shift, 5.0, 15.0)
    @test @inferred QOB.eval_coefficients(op_block_shift, 4.9) == (0.0, 0.0)
    @test @inferred QOB.eval_coefficients(op_block_shift, 5.0) == (1.0, -5.0)
    @test @inferred QOB.eval_coefficients(op_block_shift, 20.0) == (0.0, 0.0)

    op_stretch_block_shift = time_stretch(op_block_shift, 2.0)
    @test @inferred QOB.eval_coefficients(op_stretch_block_shift, 9.9) == (0.0, 0.0)
    @test @inferred QOB.eval_coefficients(op_stretch_block_shift, 10.0) == (1.0, -5.0)
    @test @inferred QOB.eval_coefficients(op_stretch_block_shift, 30.0-√eps()) == (1.0, 5.0-0.5*√eps())
    @test @inferred QOB.eval_coefficients(op_stretch_block_shift, 30.0) == (0.0, 0.0)

    op_block = time_restrict(op, 15.0)
    @test @inferred QOB.eval_coefficients(op_block, -1.0) == (0.0, 0.0)
    @test @inferred QOB.eval_coefficients(op_block, 5.0) == (1.0, 5.0)
    @test @inferred QOB.eval_coefficients(op_block, 20.0) == (0.0, 0.0)

    op_stretch = time_stretch(op, 2.0)
    @test @inferred QOB.eval_coefficients(op_stretch, 5.0) == (1.0, 2.5)
end
end