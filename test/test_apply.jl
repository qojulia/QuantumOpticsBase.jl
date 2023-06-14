using QuantumOpticsBase
using QuantumInterface
using Test

@testset "apply" begin
    
_b2 = SpinBasis(1//2)
_l0 = spinup(_b2)
_l1 = spindown(_b2)
_x = sigmax(_b2)
_y = sigmay(_b2)


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
sOp2 = spre(op)

try
    QuantumOpticsBase.apply!(st, 1, sOp2)
catch e
    @test typeof(e) <: QuantumInterface.IncompatibleBases
end

end #testset