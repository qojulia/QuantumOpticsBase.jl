function test_basic_operations(AT, synchronize)
    """Test basic quantum operations on GPU arrays."""
    cache = AllocCache()
    
    for n in test_sizes
        for r in 1:round_count
            @cached cache begin
                
                @testset "Basic Operations - Size $n" begin
                    # Create test bases
                    b1 = GenericBasis(n)
                    b2 = GenericBasis(n)
                    
                    # Test operator creation and adaptation
                    cpu_op, gpu_op = create_test_operator(b1, b2, AT)
                    
                    @test typeof(gpu_op.data) <: AT
                    @test verify_gpu_result(cpu_op, gpu_op)
                    
                    # Test ket creation and adaptation  
                    cpu_ket, gpu_ket = create_test_ket(b1, AT)
                    
                    @test typeof(gpu_ket.data) <: AT
                    @test verify_gpu_result(cpu_ket, gpu_ket)
                    
                    # Test bra creation and adaptation
                    cpu_bra, gpu_bra = create_test_bra(b1, AT)
                    
                    @test typeof(gpu_bra.data) <: AT
                    @test verify_gpu_result(cpu_bra, gpu_bra)
                    
                    # Test operator multiplication (both on GPU)
                    cpu_op2, gpu_op2 = create_test_operator(b2, b1, AT)
                    
                    cpu_result = cpu_op * cpu_op2
                    gpu_result = gpu_op * gpu_op2
                    synchronize()
                    
                    @test verify_gpu_result(cpu_result, gpu_result)
                    
                    # Test adjoint/dagger operation
                    cpu_dag = dagger(cpu_op)
                    gpu_dag = dagger(gpu_op)
                    synchronize()
                    
                    @test verify_gpu_result(cpu_dag, gpu_dag)
                    
                    # Test trace operation
                    if n == length(b2)  # Square matrices only
                        cpu_trace = tr(cpu_op)
                        gpu_trace = tr(gpu_op)
                        synchronize()
                        
                        @test isapprox(cpu_trace, gpu_trace, atol=GPU_TOL)
                    end
                    
                    # Test norm operations
                    cpu_norm = norm(cpu_ket)
                    gpu_norm = norm(gpu_ket)
                    synchronize()
                    
                    @test isapprox(cpu_norm, gpu_norm, atol=GPU_TOL)
                end
                
            end
        end
    end
end
