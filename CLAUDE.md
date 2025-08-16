# QuantumOpticsBase.jl

QuantumOpticsBase.jl provides the base functionality for QuantumOptics.jl. It implements fundamental types such as different bases, states and operators defined on these bases, and core operations (such as multiplication) on these states/operators.

## Project Structure

- `src/` - Main source code
  - `QuantumOpticsBase.jl` - Main module file
  - `bases.jl` - Different quantum bases (Fock, Spin, NLevel, etc.)
  - `states.jl` - Quantum state implementations
  - `operators*.jl` - Various operator implementations (dense, sparse, lazy)
  - `metrics.jl` - Distance metrics and measurements
  - `phasespace.jl` - Phase space functions
  - `transformations.jl` - Basis transformations
- `test/` - Comprehensive test suite
- `docs/` - Documentation source files

## Development Commands

### Running Tests
```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run specific test file  
julia --project=. test/runtests.jl

# Run with multiple threads
julia --project=. -t auto test/runtests.jl
```

### Building Documentation
```bash
# Install documentation dependencies
julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"

# Build documentation
julia --project=docs docs/make.jl
```

### Package Management
```bash
# Instantiate project dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Update dependencies
julia --project=. -e "using Pkg; Pkg.update()"

# Check package status
julia --project=. -e "using Pkg; Pkg.status()"
```

## Testing Information

The test suite uses TestItemRunner and includes:

- Unit tests for all quantum bases (Fock, Spin, NLevel, Charge, Particle)
- Operator tests (dense, sparse, lazy implementations)
- State manipulation and transformation tests
- Metrics and measurement tests
- Integration tests with other quantum packages
- Code quality tests (Aqua.jl, JET.jl, doctests)

Special test configurations:
- JET tests run when `JET_TEST=true` environment variable is set
- Aqua and doctest checks require Julia 1.10+

## Key Dependencies

- `QuantumInterface.jl` - Provides common quantum computing interfaces
- `LinearAlgebra` - Core linear algebra operations
- `SparseArrays` - Sparse matrix implementations
- `FFTW` - Fast Fourier transforms for phase space calculations
- `FastExpm` - Efficient matrix exponentials

## Development Notes

- Minimum Julia version: 1.10
- Uses semantic versioning
- Extensive test coverage with multiple CI platforms
- Documentation auto-deploys on releases
- Compatible with GPU acceleration through extensions

## Related Packages

- `QuantumOptics.jl` - Main package that builds on this base
- `QuantumInterface.jl` - Common interfaces
- See the @qojulia organization for the full ecosystem

## Contributing

This package follows standard Julia development practices:
- Fork and create feature branches
- Write tests for new functionality
- Ensure documentation builds successfully
- All tests must pass before merging