using Test
using QuantumOpticsBase

@testset "subspace" begin

b = FockBasis(3)

u = Ket[fockstate(b, 1), fockstate(b, 2)]
v = Ket[fockstate(b, 2), fockstate(b, 1)]
w = Ket[Ket(b, [1 + 1im, 1 - 0.5im, 0, 0] / norm([1 + 1im, 1 - 0.5im, 0, 0]))]

bu = SubspaceBasis(u)
bv = SubspaceBasis(v)
bw = SubspaceBasis(w)

T1 = projector(bu, b)
T2 = projector(bv, b)
T12 = projector(bu, bv)
Tw = projector(bw, b)

state = fockstate(b, 2)
state_1 = Ket(b, [1 + 1im, 1 - 0.5im, 0, 0] / norm([1 + 1im, 1 - 0.5im, 0, 0]))
state_u = Ket(bu, [0, 1])
state_v = Ket(bv, [1., 0])
state_w = Ket(bw, [1.0])
@test Tw * state_1 ≈ state_w

@test T1*state == state_u
@test T2*state == state_v


state_v = Ket(bv, [1, -1])
state_u = Ket(bu, [-1, 1])

@test T12*state_v == state_u

u2 = Ket[1.5*fockstate(b, 1), fockstate(b, 1) + fockstate(b, 2)]
bu2_orth = QuantumOpticsBase.orthonormalize(SubspaceBasis(u))
@test bu2_orth.basisstates == bu.basisstates

# Test sparse version
T1 = sparseprojector(bu, b)
T2 = sparseprojector(bv, b)
T12 = sparseprojector(bu, bv)

@test isa(T1, SparseOpType)

state_u = Ket(bu, [0, 1])
state_v = Ket(bv, [1., 0])

@test T1*state == state_u
@test T2*state == state_v

state_v = Ket(bv, [1, -1])
state_u = Ket(bu, [-1, 1])

@test T12*state_v == state_u

# Test errors
b2 = FockBasis(4)
@test_throws ArgumentError SubspaceBasis(b2, u)
@test_throws ArgumentError projector(bu, b2)
@test_throws ArgumentError projector(b2, bu)
b2_sub = SubspaceBasis(b2, [fockstate(b2, 1)])
@test_throws ArgumentError projector(bu, b2_sub)

end # testset
