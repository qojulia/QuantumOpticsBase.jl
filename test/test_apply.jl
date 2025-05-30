@testitem "apply" begin

using QuantumOpticsBase
using QuantumInterface
using QuantumOpticsBase: apply!

_b2 = SpinBasis(1//2)
_l0 = spinup(_b2)
_l1 = spindown(_b2)
_x = sigmax(_b2)
_y = sigmay(_b2)
_z = sigmaz(_b2)

@test apply!(_l0⊗_l1, 1, _x) ≈ _l1⊗_l1
@test apply!(_x, 1, _y) ≈ _x

# Test Operator with IncompatibleBases
_l01 = _l0⊗_l1
op = projector(_l0, _l01')
@test_throws QuantumInterface.IncompatibleBases apply!(_l0, 1, op)
@test_throws QuantumInterface.IncompatibleBases apply!(_x, 1, op)

# Test SuperOperator
bf1 = FockBasis(1)
sOp = spre(create(bf1))
st = coherentstate(bf1, 1.4)
@test apply!(st, 1, sOp).data ≈ (sOp*projector(st)).data

# Test SuperOperator with IncompatibleBases
bf2 = FockBasis(2)
op = create(bf2)
sOp2 = spre(op)
@test_throws ArgumentError apply!(st, 1, sOp2)

# test CNOT₂₋₁
CNOT = DenseOperator(_b2⊗_b2, Complex.([1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]))
@test QuantumOpticsBase.apply!(_l0⊗_l1, [2,1], CNOT) ≈ _l1⊗_l1

# test operator permutation with 3 qubits/operators for apply!
@test QuantumOpticsBase.apply!(_l0⊗_l1⊗_l0, [2,3,1], _x⊗_y⊗_z) ≈ (_z*_l0)⊗(_x*_l1)⊗(_y*_l0)
@test QuantumOpticsBase.apply!(_x⊗_y⊗_z, [2,3,1], _x⊗_y⊗_z) ≈ (_z⊗_x⊗_y)*(_x⊗_y⊗_z)*(_z⊗_x⊗_y)'

i = identityoperator(_y)
@test QuantumOpticsBase.apply!(_x⊗_y⊗_z, [2,3], _y⊗_z) ≈ (i⊗_y⊗_z)*(_x⊗_y⊗_z)*(i⊗_y⊗_z)'

# 3-qubit/operator errors when called for applying superoperator
sOp1 = spre(create(bf1))
st1 = coherentstate(bf1, 1.4)
st2 = coherentstate(bf1, 0.3)
@test_broken apply!(st1⊗st2, [1], sOp1)

end #testset
