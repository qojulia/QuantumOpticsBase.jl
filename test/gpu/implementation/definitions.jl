# Test parameters
const test_sizes = [2, 4, 8]
const max_rows = 16
const round_count = 2
const block_sizes = fill(64, round_count)
const batch_sizes = fill(1, round_count)

# Helper function for distance comparison
D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

# Test tolerances for GPU computations
const GPU_TOL = 1e-10