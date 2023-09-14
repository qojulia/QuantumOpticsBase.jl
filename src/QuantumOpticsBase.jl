module QuantumOpticsBase

using SparseArrays, LinearAlgebra, LRUCache, Strided, UnsafeArrays, FillArrays
import LinearAlgebra: mul!, rmul!

import QuantumInterface: dagger, directsum, ⊕, dm, embed, expect, identityoperator, identitysuperoperator,
        permutesystems, projector, ptrace, reduced, tensor, ⊗, variance, apply!, basis, AbstractSuperOperator

# index helpers
import QuantumInterface: complement, remove, shiftremove, reducedindices!, check_indices, check_sortedindices, check_embed_indices

export Basis, GenericBasis, CompositeBasis, basis,
        tensor, ⊗, permutesystems, @samebases,
        #states
                StateVector, Bra, Ket, basisstate, sparsebasisstate, norm,
                dagger, normalize, normalize!,
        #operators
                AbstractOperator, DataOperator, expect, variance,
                identityoperator, ptrace, reduced, embed, dense, tr, sparse,
        #operators_dense
                Operator, DenseOperator, DenseOpType, projector, dm,
        #operators_sparse
                SparseOperator, diagonaloperator, SparseOpType, EyeOpType,
        #operators_lazysum
                LazySum,
        #operators_lazyproduct
                LazyProduct,
        #operators_lazytensor
                LazyTensor, lazytensor_use_cache, lazytensor_clear_cache,
                lazytensor_cachesize, lazytensor_disable_cache, lazytensor_enable_cache,
        #time_dependent_operators
                AbstractTimeDependentOperator, TimeDependentSum, set_time!,
                current_time, time_shift, time_stretch, time_restrict,
        #superoperators
                SuperOperator, DenseSuperOperator, DenseSuperOpType,
                SparseSuperOperator, SparseSuperOpType, spre, spost, sprepost, liouvillian,
                identitysuperoperator,
        #fock
                FockBasis, number, destroy, create,
                fockstate, coherentstate, coherentstate!,
                displace, displace_analytical, displace_analytical!,
                squeeze,
        randstate, randoperator, thermalstate, coherentthermalstate, phase_average, passive_state,
        #spin
                SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown,
        #subspace
                SubspaceBasis, projector, sparseprojector,
        #particle
                PositionBasis, MomentumBasis, samplepoints, spacing, gaussianstate,
                position, momentum, potentialoperator, transform,
        #nlevel
                NLevelBasis, transition, nlevelstate,
        #manybody
                ManyBodyBasis, fermionstates, bosonstates,
                manybodyoperator, onebodyexpect,
        #metrics
                tracenorm, tracenorm_h, tracenorm_nh,
                tracedistance, tracedistance_h, tracedistance_nh,
                entropy_vn, entropy_renyi, fidelity, ptranspose, PPT,
                negativity, logarithmic_negativity, entanglement_entropy,
        PauliBasis, PauliTransferMatrix, DensePauliTransferMatrix,
                ChiMatrix, DenseChiMatrix, avg_gate_fidelity,
        SumBasis, directsum, ⊕, LazyDirectSum, getblock, setblock!,
        qfunc, wigner, coherentspinstate, qfuncsu2, wignersu2
        #apply
                apply!

include("bases.jl")
include("states.jl")
include("operators.jl")
include("operators_dense.jl")
include("sparsematrix.jl")
include("operators_sparse.jl")
include("operators_lazysum.jl")
include("operators_lazyproduct.jl")
include("operators_lazytensor.jl")
include("time_dependent_operator.jl")
include("superoperators.jl")
include("spin.jl")
include("fock.jl")
include("state_definitions.jl")
include("subspace.jl")
include("particle.jl")
include("nlevel.jl")
include("manybody.jl")
include("transformations.jl")
include("pauli.jl")
include("metrics.jl")
include("spinors.jl")
include("phasespace.jl")
include("printing.jl")
include("apply.jl")

end # module
