using QuantumOpticsBase
using Test
using Random

@testset "spinors" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊕b2a⊕b3a
b_r = b1b⊕b2b⊕b3b

op1a = randoperator(b1a, b1b)
op1b = randoperator(b1a, b1b)
op2a = randoperator(b2a, b2b)
op2b = randoperator(b2a, b2b)
op3a = randoperator(b3a, b3b)
op123 = op1a ⊕ op2a ⊕ op3a
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
@test 1e-13 > D((op1a ⊕ op2a) ⊕ op3a, op1a ⊕ (op2a ⊕ op3a))
@test 1e-13 > D(op1a ⊕ op2a ⊕ op3a, op1a ⊕ (op2a ⊕ op3a))

# Mixed-product property
@test 1e-13 > D((op1a ⊕ op2a) * dagger(op1b ⊕ op2b), (op1a*dagger(op1b)) ⊕ (op2a*dagger(op2b)))

# Transpose
@test 1e-13 > D(dagger(op1a ⊕ op2a), dagger(op1a) ⊕ dagger(op2a))
@test 1e-13 > D(dagger(op1a ⊕ op2a), dagger(op1a) ⊕ dagger(op2a))

# Sparse version
@test isa(sparse(op1a)⊕sparse(op2a)⊕sparse(op3a), SparseOpType)

# Test lazy implementation
L = LazyDirectSum(op1a,op2a)
@test 1e-13 > D(op1a⊕op2a, L)
@test 1e-13 > D(2*(op1a⊕op2a), L+L)
@test isa(L ⊕ op3a, LazyDirectSum)
@test 1e-13 > D(op1a ⊕ op2a ⊕ op3a, L ⊕ op3a)
@test 1e-13 > D((op1a⊕op2a)*dagger(op1a⊕op2a), L*L')


### Test example

# Define x-space
Nunitcells=2
Lx=Nunitcells/2
x_min=0
x_max = Lx
x_steps = 4

# Define y-space
Ly=Nunitcells/2
y_min = 0
y_max = Ly
y_steps = 8

# Define Bases
b_x = PositionBasis(x_min, x_max, x_steps)
b_y = PositionBasis(y_min, y_max, y_steps)
bcomp_x = b_x ⊗ b_y
b_sum = bcomp_x ⊕ bcomp_x

b_mom = MomentumBasis(b_x)
b_mom_y = MomentumBasis(b_y)
bcomp_p = MomentumBasis(b_x) ⊗ MomentumBasis(b_y)
b_sum_p = bcomp_p ⊕ bcomp_p

# Define Operators
H_up = randoperator(bcomp_x)
H_down = randoperator(b_x⊗b_y)
Hkin_x = randoperator(bcomp_x)
H = (H_up ⊕ H_down)

# Off-diagonal blocks - assign by hand
Ω_R = randoperator(bcomp_x)
nn = length(H_up.basis_l)
H.data[1:nn,nn+1:end] = Ω_R.data
H.data[nn+1:end,1:nn] = Ω_R.data'

H_h_upperline = hcat(Matrix(H_up.data),Matrix(Ω_R.data))
H_h_lowerline = hcat(Matrix(Ω_R.data'),Matrix(H_down.data))
H_matr=vcat(H_h_upperline,H_h_lowerline)
@test H.data ≈ H_matr
@test getblock(H, 1,2) == Ω_R

# States
ψ0_x=randstate(b_x)
ψ0_y=randstate(b_y)
ψ0_up=ψ0_x⊗ψ0_y
# Build with directsum (\oplus)
ψ0=(ψ0_x⊗ψ0_y)⊕(ψ0_x⊗ψ0_y)

# Test initial state
@test ψ0.data ≈ [ψ0_up.data;ψ0_up.data]
@test getblock(ψ0, 1) == ψ0_up == getblock(ψ0, 2)

# Test FFTs
Txp = tensor(transform(b_x, b_mom),transform(b_y, b_mom_y)) # "normal" FFT operators as implemented
Tpx = Txp'
ψ0_p1 = (Tpx*ψ0_up)⊕(Tpx*ψ0_up) # Build with usual FFTs
ψ0_p2 = LazyDirectSum(Tpx, Tpx)*ψ0
@test ψ0_p1.data ≈ ψ0_p2.data

# Test off-diagonal blocks
H_test = copy(H)
setblock!(H_test, Ω_R, 1, 2)
setblock!(H_test, Ω_R', 2, 1)
@test H == H_test

# Lazy formulation
Txp_tot = LazyDirectSum(Txp, Txp)
Tpx_tot = LazyDirectSum(Tpx, Tpx)
Hkin_p = Tpx*Hkin_x*Txp
Hkin_tot = LazyDirectSum(Hkin_p, Hkin_p)
H_lazy = LazySum(H, LazyProduct(Txp_tot, Hkin_tot, Tpx_tot))

Hkin_x_tot = Txp_tot*Hkin_tot*Tpx_tot
@test dense(H_lazy).data ≈ dense(dense(H) + dense(Hkin_x_tot)).data
@test (H_lazy*ψ0).data ≈ (H*ψ0 + Hkin_x_tot*ψ0).data

# # Test time evolution
# tout, ψt = timeevolution.schroedinger([0.0:0.1:1.0;], ψ0, H)
# tout, ψt_lazy = timeevolution.schroedinger([0.0:0.1:1.0;], ψ0, H_lazy)
#
# for i=1:length(tout)
#     @test ψt[i].data ≈ ψt_lazy[i].data
# end

end # testset
