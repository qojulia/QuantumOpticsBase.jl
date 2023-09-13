using QuantumOpticsBase
using QuantumInterface
using Test

@testset "apply" begin
    
_b2 = SpinBasis(1//2)
_l0 = spinup(_b2)
_l1 = spindown(_b2)
_x = sigmax(_b2)
_y = sigmay(_b2)
_z = sigmaz(_b2)

@test QuantumOpticsBase.apply!(_l0⊗_l1, 1, _x) == _l1⊗_l1
@test QuantumOpticsBase.apply!(_x, 1, _y) == _x

# Test Operator with IncompatibleBases
_l01 = _l0⊗_l1
op = projector(_l0, _l01')

try
    QuantumOpticsBase.apply!(_l0, 1, op)
catch e
    @test typeof(e) <: QuantumInterface.IncompatibleBases
end

try
    QuantumOpticsBase.apply!(_x, 1, op)
catch e
    @test typeof(e) <: QuantumInterface.IncompatibleBases
end

# Test SuperOperator
sOp = spre(create(FockBasis(1)))
st = coherentstate(FockBasis(1), 1.4)

@test QuantumOpticsBase.apply!(st, 1, sOp).data == (sOp*projector(st)).data

# Test SuperOperator with IncompatibleBases
b1 = FockBasis(1)
b2 = FockBasis(2)

k1 = coherentstate(b1, 0.39)
k2 = coherentstate(b2, 1.4)
op = projector(k1, k2')

try
    sOp2 = spre(op)
    QuantumOpticsBase.apply!(st, 1, sOp2)
catch e
    @test typeof(e) <: ArgumentError
    #@test typeof(e) <: QuantumInterface.IncompatibleBases
end

#test CNOT₂₋₁
CNOT = DenseOperator(_b2⊗_b2, Complex.([1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]))
@test QuantumOpticsBase.apply!(_l0⊗_l1, [2,1], CNOT) == _l1⊗_l1

#test operator permutation with 3 qubits/operators for apply!

# 3-qubit operator permutation
@test QuantumOpticsBase.apply!(_l0⊗_l1⊗_l0, [2,3,1], _x⊗_y⊗_z) ≈ (_y*_l0)⊗(_z*_l1)⊗(_x*_l0)

# 3-operator operator permutation
@test QuantumOpticsBase.apply!(_x⊗_y⊗_z, [2,3,1], _x⊗_y⊗_z) ≈ (_y⊗_z⊗_x)*(_x⊗_y⊗_z)*(_y⊗_z⊗_x)'

#3-qubit/operator errors when called for applying superoperator
sOp1 = spre(create(FockBasis(1)))
sOp2 = spost(create(FockBasis(1)))
st1 = coherentstate(FockBasis(1), 1.4)
st2 = coherentstate(FockBasis(1), 0.3)
st3 = coherentstate(FockBasis(1), 0.8)

@test_throws "Applying SuperOperator to multiple qubits/operators is not supported currently, due to missing tensor product method for SuperOperators" QuantumOpticsBase.apply!(st1⊗st2⊗st3, [2,3,1], sOp1)

end #testset
