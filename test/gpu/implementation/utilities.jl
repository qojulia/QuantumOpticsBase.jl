# Utility functions for GPU testing

function create_test_operator(basis_l, basis_r, AT)
    """Create a test operator and adapt it to the specified array type."""
    data = rand(ComplexF64, length(basis_l), length(basis_r))
    cpu_op = DenseOperator(basis_l, basis_r, data)
    gpu_op = Adapt.adapt(AT, cpu_op)
    return cpu_op, gpu_op
end

function create_test_ket(basis, AT)
    """Create a test ket and adapt it to the specified array type."""
    data = rand(ComplexF64, length(basis))
    normalize!(data)
    cpu_ket = Ket(basis, data)
    gpu_ket = Adapt.adapt(AT, cpu_ket)
    return cpu_ket, gpu_ket
end

function create_test_bra(basis, AT)
    """Create a test bra and adapt it to the specified array type."""
    data = rand(ComplexF64, length(basis))
    normalize!(data)
    cpu_bra = Bra(basis, data)
    gpu_bra = Adapt.adapt(AT, cpu_bra)
    return cpu_bra, gpu_bra
end

function verify_gpu_result(cpu_result, gpu_result, tolerance=GPU_TOL)
    """Verify that GPU computation matches CPU result within tolerance."""
    cpu_data = cpu_result isa AbstractOperator ? cpu_result.data : cpu_result.data
    gpu_data_cpu = Array(gpu_result isa AbstractOperator ? gpu_result.data : gpu_result.data)
    return isapprox(cpu_data, gpu_data_cpu, atol=tolerance)
end
