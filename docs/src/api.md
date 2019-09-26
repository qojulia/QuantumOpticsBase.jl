# API


## [Types](@id API: Quantum objects types)

* General basis types. Specialized bases can be found in the section [API: Quantum-systems](@ref).

```@docs
Basis
```

```@docs
GenericBasis
```

```@docs
CompositeBasis
```

* States

```@docs
StateVector
```

```@docs
Bra
```

```@docs
Ket
```

* General purpose QuantumOpticsBase. A few more specialized operators are implemented in [API: Quantum-systems](@ref).

```@docs
AbstractOperator
```

```@docs
DataOperator
```

```@docs
DenseOperator
```

```@docs
SparseOperator
```

```@docs
LazyTensor
```

```@docs
LazySum
```

```@docs
LazyProduct
```


* Super operators:

```@docs
SuperOperator
```

```@docs
DenseSuperOperator
```

```@docs
SparseSuperOperator
```


### [Functions](@id API: Quantum objects functions)

* Functions to generate general states, operators and super-operators

```@docs
basisstate
```

```@docs
identityoperator
```

```@docs
diagonaloperator
```

```@docs
randoperator
```

```@docs
spre
```

```@docs
spost
```

```@docs
liouvillian
```

* As far as it makes sense the same functions are implemented for bases, states, operators and superQuantumOpticsBase.

```@docs
QuantumOpticsBase.samebases
```

```@docs
QuantumOpticsBase.check_samebases
```

```@docs
QuantumOpticsBase.multiplicable
```

```@docs
QuantumOpticsBase.check_multiplicable
```

```@docs
QuantumOpticsBase.basis
```

```@docs
dagger
```

```@docs
tensor
```

```@docs
projector(a::Ket, b::Bra)
projector(a::Ket)
projector(a::Bra)
```

```@docs
dm
```

```@docs
QuantumOpticsBase.norm(x::StateVector)
```

```@docs
tr
```

```@docs
ptrace
```

```@docs
normalize(x::StateVector)
normalize(op::AbstractOperator)
```

```@docs
normalize!(x::StateVector)
normalize!(op::AbstractOperator)
```

```@docs
expect
```

```@docs
variance
```

```@docs
embed
```

```@docs
permutesystems
```

```@docs
exp(op::AbstractOperator)
exp(op::DenseSuperOperator)
```

```@docs
QuantumOpticsBase.gemv!
```

```@docs
QuantumOpticsBase.gemm!
```

* Conversion of operators

```@docs
dense
```

```@docs
sparse(::AbstractOperator)
```

### [Exceptions](@id API: Quantum objects exceptions)

```@docs
QuantumOpticsBase.IncompatibleBases
```


## [Quantum systems](@id API: Quantum systems)

### [Fock](@id API: Fock)

```@docs
FockBasis
```

```@docs
number(::FockBasis)
```

```@docs
destroy(::FockBasis)
```

```@docs
create(::FockBasis)
```

```@docs
displace
```

```@docs
fockstate
```

```@docs
coherentstate
```

```@docs
coherentstate!
```


### [N-level](@id API: N-level)

```@docs
NLevelBasis
```

```@docs
transition(::NLevelBasis, ::Int, ::Int)
```

```@docs
nlevelstate
```

### [Spin](@id API: Spin)

```@docs
SpinBasis
```

```@docs
sigmax
```

```@docs
sigmay
```

```@docs
sigmaz
```

```@docs
sigmap
```

```@docs
sigmam
```

```@docs
spinup
```

```@docs
spindown
```

### [Particle](@id API: Particle)

```@docs
PositionBasis
```

```@docs
MomentumBasis
```

```@docs
spacing
```

```@docs
samplepoints
```

```@docs
position(b::PositionBasis)
position(b::MomentumBasis)
```

```@docs
momentum(b::PositionBasis)
momentum(b::MomentumBasis)
```

```@docs
potentialoperator
```

```@docs
gaussianstate
```

```@docs
QuantumOpticsBase.FFTOperator
```

```@docs
QuantumOpticsBase.FFTOperators
```

```@docs
QuantumOpticsBase.FFTKets
```

```@docs
transform
```

### [Subspace bases](@id API: Subspace bases)

```@docs
SubspaceBasis
```

```@docs
QuantumOpticsBase.orthonormalize
```

```@docs
projector(b1::SubspaceBasis, b2::SubspaceBasis)
```

### [Many-body](@id API: Many-body)

```@docs
ManyBodyBasis
```

```@docs
fermionstates
```

```@docs
bosonstates
```

```@docs
number(::ManyBodyBasis, ::Int)
number(::ManyBodyBasis)
```

```@docs
destroy(::ManyBodyBasis, ::Int)
```

```@docs
create(::ManyBodyBasis, ::Int)
```

```@docs
transition(::ManyBodyBasis, ::Int, ::Int)
```

```@docs
manybodyoperator
```

```@docs
onebodyexpect
```


## [Metrics](@id API: Metrics)

```@docs
tracenorm
```

```@docs
tracenorm_h
```

```@docs
tracenorm_nh
```

```@docs
tracedistance
```

```@docs
tracedistance_h
```

```@docs
tracedistance_nh
```

```@docs
entropy_vn
```

```@docs
fidelity
```

```@docs
ptranspose
```

```@docs
PPT
```

```@docs
negativity
```

```@docs
logarithmic_negativity
```

## [State definitions](@id API: State definitions)

```@docs
randstate
```

```@docs
thermalstate
```

```@docs
coherentthermalstate
```

```@docs
phase_average
```

```@docs
passive_state
```

## [Pauli](@id API: Pauli)

```@docs
PauliBasis
```

```@docs
PauliTransferMatrix
```

```@docs
DensePauliTransferMatrix
```

```@docs
ChiMatrix
```

```@docs
DenseChiMatrix
```

## [Printing](@id API: Printing)

```@docs
QuantumOpticsBase.set_printing
```
