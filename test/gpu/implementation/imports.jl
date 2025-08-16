# GPU backend imports
import Adapt
using QuantumOpticsBase
using LinearAlgebra, Random, Test
using SparseArrays

# For memory management in GPU tests
using GPUArrays: AllocCache, @cached, unsafe_free!