@testitem "test_sciml_lazy_operators" begin
using Test
using LinearAlgebra, Random, SparseArrays
using QuantumOpticsBase
using SciMLOperators

@testset "SciMLOperators-backed lazy operators" begin
    Random.seed!(522)

    state_distance(a::StateVector, b::StateVector) = norm(a - b)
    op_distance(a::AbstractOperator, b::AbstractOperator) = abs(tracedistance_nh(dense(a), dense(b)))

    b0 = SpinBasis(1//2)
    b = tensor(b0, b0, b0, b0)
    sx = sigmax(b0)
    sz = sigmaz(b0)

    current = LazySum(
        LazyTensor(b, 1, sx),
        LazyTensor(b, 2, sz),
        LazyProduct(LazyTensor(b, 3, sx), LazyTensor(b, 4, sz)),
    )
    sciml = sciml_lazy_operator(current)
    psi = Ket(b, randn(ComplexF64, length(b)))

    @test op_distance(sciml, current) < 1e-12
    @test state_distance(sciml * psi, current * psi) < 1e-12

    out_current = Ket(b)
    out_sciml = Ket(b)
    mul!(out_current, current, psi, 0.25 - 0.5im, 0)
    mul!(out_sciml, sciml, psi, 0.25 - 0.5im, 0)
    @test state_distance(out_sciml, out_current) < 1e-12

    cached = cache_sciml_lazy_operator(sciml, psi)
    out_cached = Ket(b)
    mul!(out_cached, cached, psi, 1, 0)
    @test state_distance(out_cached, current * psi) < 1e-12

    a = LazyTensor(b, 1, sx)
    c = LazyTensor(b, 2, sz)
    a_sciml = sciml_lazy_operator(a)
    c_sciml = sciml_lazy_operator(c)

    @test size(a_sciml) == size(a)
    @test size(a_sciml, 1) == size(a, 1)
    @test size(a_sciml, 2) == size(a, 2)
    @test eltype(a_sciml) <: Number
    @test op_distance(copy(a_sciml), a) < 1e-12
    @test op_distance(sparse(a_sciml), sparse(a)) < 1e-12

    @test sciml_lazy_operator(a_sciml) === a_sciml
    @test op_distance(sciml_lazy_operator(a_sciml; cache=psi), a) < 1e-12
    @test op_distance(sciml_lazy_operator(a; cache=psi), a) < 1e-12
    @test op_distance(cache_sciml_lazy_operator(a, psi), a) < 1e-12

    @test op_distance(2a_sciml + c_sciml / 3, 2a + c / 3) < 1e-12
    @test op_distance(a_sciml + c, a + c) < 1e-12
    @test op_distance(a + c_sciml, a + c) < 1e-12
    @test op_distance(a_sciml - c, a - c) < 1e-12
    @test op_distance(a - c_sciml, a - c) < 1e-12
    @test op_distance(a_sciml + dense(c), a + dense(c)) < 1e-12
    @test op_distance(dense(a) + c_sciml, dense(a) + c) < 1e-12
    @test op_distance(a_sciml - dense(c), a - dense(c)) < 1e-12
    @test op_distance(dense(a) - c_sciml, dense(a) - c) < 1e-12
    @test op_distance(-a_sciml, -a) < 1e-12
    @test op_distance(a_sciml * 2, a * 2) < 1e-12
    @test op_distance(2 * a_sciml, 2 * a) < 1e-12
    @test op_distance(a_sciml * c_sciml, a * c) < 1e-12
    @test op_distance(a_sciml * c, a * c) < 1e-12
    @test op_distance(a * c_sciml, a * c) < 1e-12
    @test op_distance(a_sciml * dense(c), a * dense(c)) < 1e-12
    @test op_distance(dense(a) * c_sciml, dense(a) * c) < 1e-12
    @test op_distance(dagger(a_sciml), dagger(a)) < 1e-12
    @test isapprox(dense(transpose(a_sciml)).data, transpose(dense(a).data))
    @test op_distance(dense(cached), current) < 1e-12

    empty_sum = LazySum(ComplexF64, b, b)
    @test op_distance(sciml_lazy_operator(empty_sum), empty_sum) < 1e-12

    three_product = LazyProduct(
        LazyTensor(b, 1, sx),
        LazyTensor(b, 2, sz),
        LazyTensor(b, 3, sx),
    )
    @test op_distance(sciml_lazy_operator(three_product), three_product) < 1e-12
    @test op_distance(sciml_lazy_operator(LazyProduct(dense(a), dense(c))), dense(a) * dense(c)) < 1e-12

    direct_sum = LazyDirectSum(sx, sz)
    direct_sum_sciml = sciml_lazy_operator(direct_sum)
    @test op_distance(direct_sum_sciml, direct_sum) < 1e-12
    @test op_distance(direct_sum_sciml + direct_sum, direct_sum + direct_sum) < 1e-12
    @test op_distance(direct_sum + direct_sum_sciml, direct_sum + direct_sum) < 1e-12
    @test op_distance(direct_sum_sciml - direct_sum, dense(direct_sum) - dense(direct_sum)) < 1e-12
    @test op_distance(direct_sum - direct_sum_sciml, dense(direct_sum) - dense(direct_sum)) < 1e-12
    @test op_distance(direct_sum_sciml * direct_sum, direct_sum * direct_sum) < 1e-12
    @test op_distance(direct_sum * direct_sum_sciml, direct_sum * direct_sum) < 1e-12

    bra = dagger(psi)
    bra_current = Bra(b)
    bra_sciml = Bra(b)
    mul!(bra_current, bra, current, 0.5 + 0.25im, 0)
    mul!(bra_sciml, bra, sciml, 0.5 + 0.25im, 0)
    @test norm(bra_sciml.data - bra_current.data) < 1e-12

    bra_fast_current = Bra(b)
    bra_fast_sciml = Bra(b)
    mul!(bra_fast_current, bra, current, 1, 0)
    mul!(bra_fast_sciml, bra, sciml, 1, 0)
    @test norm(bra_fast_sciml.data - bra_fast_current.data) < 1e-12

    b_other = SpinBasis(1)
    other_sciml = sciml_lazy_operator(sigmax(b_other))
    @test_throws QuantumOpticsBase.IncompatibleBases a_sciml + other_sciml
    @test_throws QuantumOpticsBase.IncompatibleBases a_sciml * other_sciml

    ext = Base.get_extension(QuantumOpticsBase, :QuantumOpticsBaseSciMLOperatorsExt)
    small_operator = SciMLOperators.MatrixOperator(zeros(ComplexF64, 2, 2))
    @test_throws DimensionMismatch ext.SciMLLazyOperator(b, b, small_operator)

    b1l = GenericBasis(2)
    b1r = GenericBasis(4)
    b2 = GenericBasis(3)
    bl = tensor(b1l, b2)
    br = tensor(b1r, b2)
    local_op = randoperator(b2)
    rectangular = LazyTensor(bl, br, [2], (local_op,), 0.7)
    rectangular_sciml = sciml_lazy_operator(rectangular)
    psi_rectangular = Ket(br, randn(ComplexF64, length(br)))

    @test op_distance(rectangular_sciml, rectangular) < 1e-12
    @test state_distance(rectangular_sciml * psi_rectangular, rectangular * psi_rectangular) < 1e-12
end
end
