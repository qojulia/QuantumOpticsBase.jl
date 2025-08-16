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

# Run with specific GPU backend
CUDA_TEST=true julia --project=. -e "using Pkg; Pkg.test()"
AMDGPU_TEST=true julia --project=. -e "using Pkg; Pkg.test()"
OpenCL_TEST=true julia --project=. -e "using Pkg; Pkg.test()"
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
- Compatible with GPU acceleration through Adapt.jl (converting main memory arrays to GPU arrays)

## Related Packages

- `QuantumOptics.jl` - Main package that builds on this base
- `QuantumInterface.jl` - Common interfaces
- See the @qojulia organization for the full ecosystem

## Code Formatting

### Removing Trailing Whitespaces
Before committing, ensure there are no trailing whitespaces in Julia files:

```bash
# Remove trailing whitespaces from all .jl files (requires gnu tools)
find . -type f -name '*.jl' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
```

### Ensuring Files End with Newlines
Ensure all Julia files end with a newline to avoid misbehaving CLI tools:

```bash
# Add newline to end of all .jl files that don't have one
find . -type f -name '*.jl' -exec sed -i '$a\' {} \+
```

### General Formatting Guidelines
- Use 4 spaces for indentation (no tabs)
- Remove trailing whitespaces from all lines
- Ensure files end with a single newline
- Follow Julia standard naming conventions
- Keep lines under 100 characters when reasonable

## Contributing

This package follows standard Julia development practices:
- **Always pull latest changes first**: Before creating any new feature or starting work, ensure you have the latest version by running `git pull origin master` (or `git pull origin main`)
- **Pull before continuing work**: Other maintainers might have modified the branch you are working on. Always call `git pull` before continuing work on an existing branch
- Fork and create feature branches
- Write tests for new functionality
- Ensure documentation builds successfully
- Follow code formatting guidelines above
- All tests must pass before merging