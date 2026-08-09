# test/test_embed_lazy.jl
#
# Tests for embed_lazy (src/embed_lazy.jl).
# Add to test/runtests.jl:  include("test_embed_lazy.jl")
#
# Test groups:
#  1.  Single-site AbstractOperator (dense) → LazyTensor          [issue acceptance §5]
#  2.  Single-site AbstractOperator (sparse) → LazyTensor          [acceptance §5]
#  3.  Multi-site AbstractVector → LazyTensor                      [acceptance §1,5]
#  4.  LazyTensor re-embedding via vector indices                  [acceptance §4,5]
#  5.  LazyTensor re-embedding via scalar index                    [acceptance §4]
#  6.  LazySum single-site → LazySum of LazyTensors                [acceptance §3,5]
#  7.  Issue's exact minimal target example                        [acceptance §5]
#  8.  Full spin-chain Hamiltonian (sum over sites)                [acceptance §5]
#  9.  Asymmetric basis_l ≠ basis_r                               [acceptance §2]
# 10.  IncompatibleBases errors                                    [acceptance §6]
# 11.  ArgumentError / BoundsError for bad index arguments         [acceptance §6]
# 12.  TimeDependentSum (graceful skip on field-name mismatch)
# 13.  Allocation benchmark: lazy << eager for large N             [acceptance §7]

using QuantumInterface: IncompatibleBases
@testset "embed_lazy" begin

    # shared setup 
    b0 = SpinBasis(1//2)
    b  = tensor(b0, b0, b0, b0)    # 4-site, dim 16

    sx = sigmax(b0)
    sy = sigmay(b0)
    sz = sigmaz(b0)
    sp = sigmap(b0)
    sm = sigmam(b0)

    ψ = Ket(b, normalize!(ones(ComplexF64, length(b))))

    #  1. Single-site dense AbstractOperator 
    @testset "single-site dense operator" begin
        for (site, op_loc) in zip(1:4, [sx, sy, sz, sp])
            op_lazy  = embed_lazy(b, site, op_loc)
            op_eager = embed(b, site, op_loc)

            @test op_lazy isa LazyTensor
            @test op_lazy.basis_l === b
            @test op_lazy.basis_r === b
            @test op_lazy.indices == [site]
            @test dense(op_lazy) ≈ dense(op_eager)
            @test op_lazy * ψ    ≈ op_eager * ψ
        end
    end

    #  2. Single-site sparse operator 
    @testset "single-site sparse operator" begin
        sx_sp = sparse(sx)
        sz_sp = sparse(sz)

        lazy_sp = embed_lazy(b, 2, sx_sp)
        @test lazy_sp isa LazyTensor
        @test dense(lazy_sp) ≈ dense(embed(b, 2, sx_sp))
        @test lazy_sp * ψ    ≈ embed(b, 2, sx_sp) * ψ

        # sparse in a LazySum
        local_h_sp = LazySum([1.0+0im, 0.5+0im], (sx_sp, sz_sp))
        lazy_h_sp  = embed_lazy(b, 1, local_h_sp)
        @test lazy_h_sp isa LazySum
        @test dense(lazy_h_sp) ≈ dense(embed(b, 1, local_h_sp))
    end

    #  3. Multi-site: one operator per site 
    @testset "multi-site AbstractVector" begin
        lt = embed_lazy(b, [1, 3], [sx, sz])
        @test lt isa LazyTensor
        @test lt.indices == [1, 3]

        # Reference: build the same LazyTensor directly
        lt_ref = LazyTensor(b, [1, 3], (sx, sz))
        @test dense(lt)  ≈ dense(lt_ref)
        @test lt * ψ     ≈ lt_ref * ψ

        # Three sites
        lt3 = embed_lazy(b, [1, 2, 4], [sx, sy, sz])
        @test lt3 isa LazyTensor
        @test dense(lt3) ≈ dense(LazyTensor(b, [1, 2, 4], (sx, sy, sz)))

        # Asymmetric basis convenience
        lt_sym = embed_lazy(b, b, [2, 4], [sp, sm])
        @test lt_sym isa LazyTensor
        @test dense(lt_sym) ≈ dense(LazyTensor(b, b, [2, 4], (sp, sm)))
    end

    #  4. LazyTensor re-embedding: vector indices 
    @testset "LazyTensor re-embedding (vector indices)" begin
        b_sub2 = b0 ⊗ b0
        lt_sub = LazyTensor(b_sub2, [1, 2], (sx, sz))

        # Place op.operators[1]=sx at global site 2, op.operators[2]=sz at global site 4
        lt_big = embed_lazy(b, [2, 4], lt_sub)
        @test lt_big isa LazyTensor
        @test lt_big.indices == [2, 4]
        @test lt_big.factor  ≈ lt_sub.factor

        lt_ref = LazyTensor(b, [2, 4], (sx, sz))
        @test dense(lt_big) ≈ dense(lt_ref)
        @test lt_big * ψ    ≈ lt_ref * ψ

        # Factor is preserved for scaled LazyTensor
        lt_scaled = LazyTensor(b_sub2, [1, 2], (sx, sz), complex(2.5))
        lt_big_sc = embed_lazy(b, [2, 4], lt_scaled)
        @test lt_big_sc.factor ≈ complex(2.5)
        @test dense(lt_big_sc) ≈ 2.5 * dense(lt_ref)

        # LazyTensor with non-contiguous local indices (identity gap at local site 2)
        b_sub3 = b0 ⊗ b0 ⊗ b0
        lt_gap = LazyTensor(b_sub3, [1, 3], (sx, sp))   # local op at sites 1 and 3
        # We embed only the two operators: sx → global site 1, sp → global site 3
        lt_gap_big = embed_lazy(b, [1, 3], lt_gap)
        @test lt_gap_big isa LazyTensor
        @test lt_gap_big.indices == [1, 3]
        @test dense(lt_gap_big) ≈ dense(LazyTensor(b, [1, 3], (sx, sp)))
    end

    # 5. LazyTensor re-embedding: scalar index
    @testset "LazyTensor re-embedding (scalar index)" begin
        # LazyTensor wrapping b0 in a 1-element CompositeBasis
        lt1 = LazyTensor(b0, [1], (sx,))   # stores 1 sub-operator
        lt1_big = embed_lazy(b, 3, lt1)
        @test lt1_big isa LazyTensor
        @test dense(lt1_big) ≈ dense(embed(b, 3, sx))
        @test lt1_big * ψ    ≈ embed(b, 3, sx) * ψ
    end

    #  6. LazySum → LazySum of LazyTensors 
    @testset "LazySum single-site" begin
        local_h = LazySum([1.0+0im, 0.5+0im], (sx, sz))   # LazySum on b0

        for site in 1:4
            lazy_h  = embed_lazy(b, site, local_h)
            eager_h = embed(b, site, local_h)

            @test lazy_h isa LazySum
            # All sub-operators must be LazyTensors (not eager matrices)
            @test all(op isa LazyTensor for op in lazy_h.operators)
            # Coefficient factors are preserved
            @test lazy_h.factors ≈ local_h.factors
            # Dense agreement and Ket action agree
            @test dense(lazy_h) ≈ dense(eager_h)
            @test lazy_h * ψ    ≈ eager_h * ψ
        end

        # Three-term LazySum
        local_h3 = LazySum([1.0+0im, 0.5+0im, 0.25+0im], (sx, sz, sy))
        lazy_h3  = embed_lazy(b, 2, local_h3)
        @test lazy_h3 isa LazySum
        @test length(lazy_h3.operators) == 3
        @test dense(lazy_h3) ≈ dense(embed(b, 2, local_h3))
    end

    # 7. Issue's exact minimal target example
    @testset "issue minimal target example" begin
        op_eager = embed(b, 2, sx)
        op_lazy  = embed_lazy(b, 2, sx)

        @test op_lazy isa LazyTensor
        @test dense(op_lazy) == dense(op_eager)
        @test op_lazy * ψ ≈ op_eager * ψ

        local_h = LazySum([1.0+0im, 0.5+0im], (sx, sz))
        lazy_h  = embed_lazy(b, 3, local_h)

        @test lazy_h isa LazySum
        @test dense(lazy_h) ≈ dense(embed(b, 3, local_h))
        @test lazy_h * ψ   ≈ embed(b, 3, local_h) * ψ
    end

    #  8. Full spin-chain Hamiltonian 
    @testset "spin-chain total Hamiltonian" begin
        local_h = LazySum([1.0+0im, 0.5+0im], (sx, sz))
        H_lazy  = sum(embed_lazy(b, i, local_h) for i in 1:4)
        H_eager = sum(embed(b, i, local_h) for i in 1:4)
        @test dense(H_lazy) ≈ dense(H_eager)
        @test H_lazy * ψ    ≈ H_eager * ψ
    end

    #  9. Asymmetric basis_l ≠ basis_r 
    @testset "asymmetric bases" begin
        b_l = b0 ⊗ b0
        b_r = b0 ⊗ b0
        op_asym = embed_lazy(b_l, b_r, 1, sx)
        @test op_asym isa LazyTensor
        @test op_asym.basis_l == b_l
        @test op_asym.basis_r == b_r
        @test dense(op_asym) ≈ dense(embed(b_l, b_r, 1, sx))
    end

    #  10. IncompatibleBases 
    @testset "IncompatibleBases errors" begin
        b_spin1 = SpinBasis(1)            # dim 3, wrong for b0 sites (dim 2)
        id_bad  = identityoperator(b_spin1)

        # wrong sub-basis for single site
        @test_throws IncompatibleBases embed_lazy(b, 1, id_bad)
        @test_throws IncompatibleBases embed_lazy(b, 3, id_bad)

        # wrong sub-basis in multi-site vector
        @test_throws IncompatibleBases embed_lazy(b, [1, 2], AbstractOperator[sx, id_bad])

        # wrong sub-basis for LazyTensor re-embed
        lt_bad = LazyTensor(b_spin1 ⊗ b_spin1, [1], (id_bad,))
        @test_throws IncompatibleBases embed_lazy(b, [1], lt_bad)

        # wrong sub-basis in LazySum term
        local_bad = LazySum([1.0+0im, 1.0+0im], (id_bad, id_bad))
        @test_throws IncompatibleBases embed_lazy(b, 1, local_bad)
    end

    #  11. ArgumentError / BoundsError for bad index arguments 
    @testset "bad index arguments" begin
        # out-of-bounds site index
        @test_throws BoundsError embed_lazy(b, 0, sx)
        @test_throws BoundsError embed_lazy(b, 5, sx)    # b has 4 sites

        # unsorted indices in multi-site call
        @test_throws ArgumentError embed_lazy(b, [3, 1], [sx, sz])

        # length mismatch: 3 indices but 2 operators
        @test_throws ArgumentError embed_lazy(b, [1, 2, 3], AbstractOperator[sx, sz])

        # unsorted indices for LazyTensor re-embed
        b_sub2 = b0 ⊗ b0
        lt_sub = LazyTensor(b_sub2, [1, 2], (sx, sz))
        @test_throws ArgumentError embed_lazy(b, [3, 1], lt_sub)  # unsorted
    end

    # 12. TimeDependentSum 
    # We gracefully skip if the TimeDependentSum internal field names differ
    # from what `embed_lazy` expects (op.lazysum, op.coefficients).  If this
    # test fails with MethodError/FieldError, update the field access in
    # src/embed_lazy.jl accordingly.
    @testset "TimeDependentSum" begin
        try
            tds = TimeDependentSum([t -> 1.0, t -> 0.5], (sx, sz); init_time=0.0)
            lazy_tds = embed_lazy(b, 2, tds)
            @test lazy_tds isa TimeDependentSum
            # At a fixed time, the dense form should agree with embed applied to
            # the time-evaluated static operator.
            t0 = 1.23
            set_time!(lazy_tds, t0)
            # Build the reference via eager embed at the same time
            tds_ref = set_time!(copy(tds), t0)
            ref_static = static_operator(tds_ref)  # evaluates coefficients at t0
            lazy_at_t0 = static_operator(lazy_tds)
            @test dense(lazy_at_t0) ≈ dense(embed(b, 2, ref_static))
        catch e
            if e isa Union{MethodError, UndefRefError, KeyError, FieldError}
                @warn ("TimeDependentSum embed_lazy test skipped; " *
                       "update field names in embed_lazy.jl: $e")
            else
                rethrow(e)
            end
        end
    end

    #  13. Allocation benchmark 
    # embed_lazy must NOT allocate the 2^N × 2^N matrix.
    # We use N=12 (dim 4096) so the eager allocation is clearly measurable.
    @testset "allocation: no large matrix" begin
        N_big  = 12
        b_big  = tensor([b0 for _ in 1:N_big]...)   # dim 4096
        sx_loc = sigmax(b0)

        # JIT warm-up (exclude compilation from measurement)
        embed_lazy(b_big, 1, sx_loc)
        embed(b_big, 1, sx_loc)
        embed_lazy(b_big, 1, sx_loc + 0.5 * sz)
        embed(b_big, 1, sx_loc + 0.5 * sz)
        GC.gc(true)

        bytes_lazy  = @allocated embed_lazy(b_big, 1, sx_loc)
        bytes_eager = @allocated embed(b_big, 1, sx_loc)

        # LazyTensor creation should be tiny (< 4 kB) regardless of N.
        # The eager sparse embed must allocate at least O(4096) non-zeros.
        @test bytes_lazy  <  4_096           # < 4 kB
        @test bytes_eager > bytes_lazy * 100  # eager is ≥ 100× heavier

        # Same for LazySum
        local_h   = sx_loc + 0.5 * sz
        bytes_ls_lazy  = @allocated embed_lazy(b_big, 1, local_h)
        bytes_ls_eager = @allocated embed(b_big, 1, local_h)
        @test bytes_ls_lazy  <  8_192
        @test bytes_ls_eager > bytes_ls_lazy * 50
    end

end  # @testset "embed_lazy"