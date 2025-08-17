# GPU backend imports
import Adapt
using QuantumOpticsBase
import QuantumOpticsBase: ChoiState  # Import ChoiState for GPU tests
using LinearAlgebra, Random, Test
using SparseArrays

# For memory management in GPU tests
using GPUArrays: AllocCache, @cached, unsafe_free!
