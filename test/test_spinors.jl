@testitem "test_spinors" begin
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

# Test embedding
@test embed(b_l,b_r,1,op1a) == op1a ⊕ SparseOperator(b2a,b2b) ⊕ SparseOperator(b3a,b3b)
@test embed(b_l,b_r,[1,3],op1a ⊕ op3a) == embed(b_l,b_r,[1,3],(op1a,op3a)) == op1a ⊕ SparseOperator(b2a,b2b) ⊕ op3a

# Test with off-diagonal blocks
op = op1a ⊕ op3a
op_upper = randoperator(b1a,b3b)
setblock!(op, op_upper, 1, 2)
op_lower = randoperator(b1b,b3a)
setblock!(op, op_lower', 2, 1)

op_tot = op1a ⊕ SparseOperator(b2a,b2b) ⊕ op3a
setblock!(op_tot, op_upper, 1, 3)
setblock!(op_tot, op_lower', 3, 1)
@test op_tot == embed(b_l,b_r,[1,3],op)

# Lazy embed
@test embed(b_l,b_r,[1,2],L) == L ⊕ SparseOperator(b3a,b3b)

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
Txp = transform(bcomp_x, bcomp_p; ket_only=true) # "normal" FFT operators as implemented
Tpx = transform(bcomp_p, bcomp_x; ket_only=true)
ψ0_p1 = (Tpx*ψ0_up)⊕(Tpx*ψ0_up) # Build with usual FFTs
ψ0_p2 = (Tpx⊕Tpx)*ψ0
@test ψ0_p1.data ≈ ψ0_p2.data

# Test off-diagonal blocks
H_test = copy(H)
setblock!(H_test, Ω_R, 1, 2)
setblock!(H_test, Ω_R', 2, 1)
@test H == H_test

# Lazy formulation
Txp_tot = Txp ⊕ Txp
Tpx_tot = Tpx ⊕ Tpx
Hkin_p = Tpx*Hkin_x*Txp
Hkin_tot = Hkin_p ⊕ Hkin_p
H_lazy = LazySum(H, LazyProduct(Txp_tot, Hkin_tot, Tpx_tot))

Hkin_x_tot = Txp_tot*Hkin_tot*Tpx_tot
@test dense(H_lazy).data ≈ dense(H + Hkin_x_tot).data
@test (H_lazy*ψ0).data ≈ (H*ψ0 + Hkin_x_tot*ψ0).data

# # Test time evolution
# tout, ψt = timeevolution.schroedinger([0.0:0.1:1.0;], ψ0, H)
# tout, ψt_lazy = timeevolution.schroedinger([0.0:0.1:1.0;], ψ0, H_lazy)
#
# for i=1:length(tout)
#     @test ψt[i].data ≈ ψt_lazy[i].data
# end

end # testset
end
