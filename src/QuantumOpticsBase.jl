module QuantumOpticsBase

using SparseArrays, LinearAlgebra, LRUCache, Strided, UnsafeArrays, FillArrays
import LinearAlgebra: mul!, rmul!
import RecursiveArrayTools

import QuantumInterface: Basis, GenericBasis, CompositeBasis, basis, basis_l, basis_r, dimension, shape,
    IncompatibleBases, @compatiblebases, samebases, check_samebases,
    addible, check_addible, multiplicable, check_multiplicable, reduced, ptrace, permutesystems,
    dagger, directsum, ⊕, dm, embed, expect, identityoperator, identitysuperoperator,
    permutesystems, projector, ptrace, reduced, tensor, ⊗, variance, apply!,
    vec, unvec, super, choi, kraus, stinespring, pauli, chi, spre, spost, sprepost, liouvillian

# metrics
import QuantumInterface: entropy_vn, fidelity, logarithmic_negativity

# index helpers
import QuantumInterface: complement, remove, shiftremove, reducedindices!, check_indices, check_sortedindices, check_embed_indices

export Basis, GenericBasis, CompositeBasis, basis, basis_l, basis_r, dimension, shape,
        tensor, ⊗, permutesystems, @compatiblebases,
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
        #states_lazyket
                LazyKet,
        #time_dependent_operators
                AbstractTimeDependentOperator, TimeDependentSum, set_time!,
                current_time, time_shift, time_stretch, time_restrict, static_operator,
        #superoperators
                KetBraBasis, ChoiBasis, PauliBasis,
                vec, unvec, super, choi, kraus, stinespring, pauli, chi,
                spre, spost, sprepost, liouvillian, identitysuperoperator,
                SuperOperatorType, DenseSuperOpType, SparseSuperOpType,
        #fock
                FockBasis, number, destroy, create,
                fockstate, coherentstate, coherentstate!,
                displace, displace_analytical, displace_analytical!,
                squeeze,
        # charge
                ChargeBasis, ShiftedChargeBasis, chargestate, chargeop, expiφ, cosφ, sinφ,
        randstate, randoperator, thermalstate, coherentthermalstate, phase_average, passive_state,
        randstate_haar, randunitary_haar,
        #spin
                SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown,
        #subspace
                SubspaceBasis, projector, sparseprojector,
        #particle
                PositionBasis, MomentumBasis, samplepoints, spacing, gaussianstate,
                position, momentum, potentialoperator, transform,
        #nlevel
                NLevelBasis, transition, nlevelstate, paulix, pauliz, pauliy,
        #manybody
                ManyBodyBasis, FermionBitstring, fermionstates, bosonstates,
                manybodyoperator, onebodyexpect,
        #metrics
                tracenorm, tracenorm_h, tracenorm_nh,
                tracedistance, tracedistance_h, tracedistance_nh,
                entropy_vn, entropy_renyi, fidelity, ptranspose, PPT,
                negativity, logarithmic_negativity, entanglement_entropy,
                avg_gate_fidelity,
        SumBasis, directsum, ⊕, LazyDirectSum, getblock, setblock!,
        qfunc, wigner, coherentspinstate, qfuncsu2, wignersu2
        #apply
                apply!

include("states.jl")
include("operators.jl")
include("operators_dense.jl")
include("sparsematrix.jl")
include("operators_sparse.jl")
include("operators_lazysum.jl")
include("operators_lazyproduct.jl")
include("operators_lazytensor.jl")
include("time_dependent_operator.jl")
include("states_lazyket.jl")
include("superoperators.jl")
include("spin.jl")
include("fock.jl")
include("charge.jl")
include("state_definitions.jl")
include("subspace.jl")
include("particle.jl")
include("nlevel.jl")
include("manybody.jl")
include("transformations.jl")
#include("pauli.jl")
include("metrics.jl")
include("spinors.jl")
include("phasespace.jl")
include("printing.jl")
include("apply.jl")

end # module
