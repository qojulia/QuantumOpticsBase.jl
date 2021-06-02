using Test
using QuantumOpticsBase

@testset "printing" begin

@test sprint(show, GenericBasis([2, 3])) == "Basis(shape=[2,3])"
@test sprint(show, GenericBasis(2)) == "Basis(dim=2)"
@test sprint(show, SpinBasis(1//1)) == "Spin(1)"
@test sprint(show, SpinBasis(3//2)) == "Spin(3/2)"
@test sprint(show, FockBasis(1)) == "Fock(cutoff=1)"
@test sprint(show, NLevelBasis(2)) == "NLevel(N=2)"

@test sprint(show, PositionBasis(-4, 4, 10)) == "Position(xmin=-4, xmax=4, N=10)"
@test sprint(show, MomentumBasis(-4, 4, 10)) == "Momentum(pmin=-4, pmax=4, N=10)"

b_fock = FockBasis(4)
states = [fockstate(b_fock, 2), coherentstate(b_fock, 0.4)]
@test sprint(show, SubspaceBasis(states)) == "Subspace(superbasis=Fock(cutoff=4), states:2)"

b_mb = ManyBodyBasis(b_fock, fermionstates(b_fock, 2))
@test sprint(show, b_mb) == "ManyBody(onebodybasis=Fock(cutoff=4), states:10)"

state = fockstate(FockBasis(2), 2)
@test sprint(show, state) == "Ket(dim=3)\n  basis: Fock(cutoff=2)\n 0.0 + 0.0im\n 0.0 + 0.0im\n 1.0 + 0.0im"
state = dagger(state)
@test sprint(show, state) == "Bra(dim=3)\n  basis: Fock(cutoff=2)\n 0.0 - 0.0im\n 0.0 - 0.0im\n 1.0 - 0.0im"

op = DenseOperator(FockBasis(1))
@test sprint(show, op) == "Operator(dim=2x2)
  basis: Fock(cutoff=1)
 0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im"

op = DenseOperator(b_fock, b_fock ⊗ SpinBasis(1//2))
@test sprint(show, op) == "Operator(dim=5x10)
  basis left:  Fock(cutoff=4)
  basis right: [Fock(cutoff=4) ⊗ Spin(1/2)]
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im"

op = SparseOperator(b_fock, b_fock ⊗ SpinBasis(1//2))
if VERSION < v"1.6.0-beta1"
@test sprint(show, op) == "Operator(dim=5x10)
  basis left:  Fock(cutoff=4)
  basis right: [Fock(cutoff=4) ⊗ Spin(1/2)]
    []"
else
@test sprint(show, op) == "Operator(dim=5x10)\n  basis left:  Fock(cutoff=4)\n  basis right: [Fock(cutoff=4) ⊗ Spin(1/2)]\n     ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅          ⋅    "
end

op = SparseOperator(b_fock)
op.data[2,2] = 1
if VERSION < v"1.6.0-beta1"
@test replace(sprint(show, op), "\t" => "  ") == "Operator(dim=5x5)
  basis: Fock(cutoff=4)\n  [2, 2]  =  1.0+0.0im"
else
@test replace(sprint(show, op), "\t" => "  ") == "Operator(dim=5x5)\n  basis: Fock(cutoff=4)\n     ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅      1.0+0.0im      ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅    \n     ⋅          ⋅          ⋅          ⋅          ⋅    "
end

op = LazySum(SparseOperator(b_fock), DenseOperator(b_fock))
@test sprint(show, op) == "LazySum(dim=5x5)
  basis: Fock(cutoff=4)
  operators: 2"

op = LazyProduct(SparseOperator(b_fock), DenseOperator(b_fock))
@test sprint(show, op) == "LazyProduct(dim=5x5)
  basis: Fock(cutoff=4)
  operators: 2"

b_fock = FockBasis(2)
b_spin = SpinBasis(1//2)
b_mb = ManyBodyBasis(b_spin, fermionstates(b_spin, 1))
op = LazyTensor(b_fock ⊗ b_mb ⊗ b_spin, [1, 3], [SparseOperator(b_fock), DenseOperator(b_spin)])
@test sprint(show, op) == "LazyTensor(dim=12x12)
  basis: [Fock(cutoff=2) ⊗ ManyBody(onebodybasis=Spin(1/2), states:2) ⊗ Spin(1/2)]
  operators: 2
  indices: [1,3]"

bx = PositionBasis(-2, 2, 4)
bp = MomentumBasis(bx)
Tpx = transform(bp, bx)
@test sprint(show, Tpx) == "FFTOperators(dim=4x4)
  basis left:  Momentum(pmin=-3.141592653589793, pmax=3.141592653589793, N=4)
  basis right: Position(xmin=-2, xmax=2, N=4)"

# Inversed tensor product ordering
QuantumOpticsBase.set_printing(standard_order=true)

n = fockstate(b_fock, 1)
@test sprint(show, n) == "Ket(dim=3)\n  basis: Fock(cutoff=2)\n 0.0 + 0.0im\n 1.0 + 0.0im\n 0.0 + 0.0im"

spin1 = spindown(b_spin)
spin2 = spinup(b_spin)
state = n ⊗ spin1 ⊗ spin2
state_data = kron(n.data, spin1.data, spin2.data)
state_data_str = sprint(Base.print_array, state_data)
@test sprint(show, state) == "Ket(dim=12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n"*state_data_str

state_data_str = sprint(Base.print_array, conj.(state_data))
@test sprint(show, dagger(state)) == "Bra(dim=12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n"*state_data_str

op = dm(state)
op_data = state_data * state_data'
type_len = if VERSION < v"1.6.0-beta1"
    length("Complex{Float64}")
else
    length("ComplexF64")
end
op_data_str1 = split(sprint(show, op_data)[type_len+2:end-1], ";")
for i=1:length(op_data_str1)
    op_data_str1[i] = join(split(op_data_str1[i]), "  ")
    op_data_str1[i] = replace(op_data_str1[i], "  +  "=>"+")
end
op_data_str = join(op_data_str1, "\n ")
@test sprint(show, op) == "Operator(dim=12x12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n "*op_data_str

op = sparse(op)
op_data = sparse(op_data)
op_data_str = sprint(show, op_data)[4:end]
# @test sprint(show, op) == "Operator(dim=12x12)
#   basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n  "*op_data_str

paulix, pauliy = sigmax(b_spin), sigmay(b_spin)
pauli = paulix ⊗ pauliy
@test sprint(show, pauli) == "Operator(dim=4x4)\n  basis: [Spin(1/2) ⊗ Spin(1/2)]\n  [4, 1]  =  0.0+1.0im\n  [3, 2]  =  0.0-1.0im\n  [2, 3]  =  0.0+1.0im\n  [1, 4]  =  0.0-1.0im"
@test sprint((io, x) -> show(IOContext(io, :limit=>true, :displaysize=>(20,80)), x), pauli ⊗ pauli) == "Operator(dim=16x16)\n  basis: [Spin(1/2) ⊗ Spin(1/2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n  [16,  1]  =  -1.0+0.0im\n  [15,  2]  =  1.0+0.0im\n  [14,  3]  =  -1.0+0.0im\n  [13,  4]  =  1.0+0.0im\n  [12,  5]  =  1.0+0.0im\n  ⋮\n  [6 , 11]  =  -1.0+0.0im\n  [5 , 12]  =  1.0+0.0im\n  [4 , 13]  =  1.0+0.0im\n  [3 , 14]  =  -1.0-0.0im\n  [2 , 15]  =  1.0+0.0im\n  [1 , 16]  =  -1.0-0.0im"
@test sprint(show, dense(pauli)) == "Operator(dim=4x4)
  basis: [Spin(1/2) ⊗ Spin(1/2)]
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0-1.0im
 0.0+0.0im  0.0+0.0im  0.0+1.0im  0.0+0.0im
 0.0+0.0im  0.0-1.0im  0.0+0.0im  0.0+0.0im
 0.0+1.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im"

hadamard = DenseOperator(b_spin, 1/sqrt(2) * Complex.([1 1; 1 -1]))
@test sprint(show, sigmax(b_spin) ⊗ hadamard) == "Operator(dim=4x4)\n  basis: [Spin(1/2) ⊗ Spin(1/2)]\n  [3, 1]  =  0.707107+0.0im\n  [4, 1]  =  0.707107+0.0im\n  [3, 2]  =  0.707107+0.0im\n  [4, 2]  =  -0.707107+0.0im\n  [1, 3]  =  0.707107+0.0im\n  [2, 3]  =  0.707107+0.0im\n  [1, 4]  =  0.707107+0.0im\n  [2, 4]  =  -0.707107+0.0im"
@test sprint((io, x) -> show(IOContext(io, :limit=>true, :displaysize=>(20,80)), x), dense(sigmax(b_spin) ⊗ hadamard ⊗ hadamard ⊗ hadamard)) ==
"Operator(dim=16x16)\n  basis: [Spin(1/2) ⊗ Spin(1/2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n      0.0+0.0im        0.0+0.0im  …   0.353553+0.0im   0.353553+0.0im\n      0.0+0.0im        0.0+0.0im      0.353553+0.0im  -0.353553+0.0im\n      0.0+0.0im        0.0+0.0im     -0.353553+0.0im  -0.353553+0.0im\n      0.0+0.0im        0.0+0.0im     -0.353553+0.0im   0.353553-0.0im\n      0.0+0.0im        0.0+0.0im     -0.353553+0.0im  -0.353553+0.0im\n      0.0+0.0im        0.0+0.0im  …  -0.353553+0.0im   0.353553-0.0im\n      0.0+0.0im        0.0+0.0im      0.353553+0.0im   0.353553+0.0im\n      0.0+0.0im        0.0+0.0im      0.353553+0.0im  -0.353553+0.0im\n 0.353553+0.0im   0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im  -0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im   0.353553+0.0im  …        0.0+0.0im        0.0+0.0im\n 0.353553+0.0im  -0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im   0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im  -0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im   0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im  -0.353553+0.0im  …        0.0+0.0im        0.0+0.0im"

@test sprint((io, x) -> show(IOContext(io, :limit=>true, :displaysize=>(16,80)), x), dense(sigmax(b_spin) ⊗ hadamard ⊗ hadamard ⊗ hadamard)) ==
"Operator(dim=16x16)\n  basis: [Spin(1/2) ⊗ Spin(1/2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n      0.0+0.0im        0.0+0.0im  …   0.353553+0.0im   0.353553+0.0im\n      0.0+0.0im        0.0+0.0im      0.353553+0.0im  -0.353553+0.0im\n      0.0+0.0im        0.0+0.0im     -0.353553+0.0im  -0.353553+0.0im\n      0.0+0.0im        0.0+0.0im     -0.353553+0.0im   0.353553-0.0im\n      0.0+0.0im        0.0+0.0im     -0.353553+0.0im  -0.353553+0.0im\n      0.0+0.0im        0.0+0.0im  …  -0.353553+0.0im   0.353553-0.0im\n         ⋮                        ⋱                            ⋮     \n 0.353553+0.0im  -0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im   0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im  -0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im   0.353553+0.0im           0.0+0.0im        0.0+0.0im\n 0.353553+0.0im  -0.353553+0.0im  …        0.0+0.0im        0.0+0.0im"

nlevel1 = basisstate(NLevelBasis(4), 1)
nlevel2 = basisstate(NLevelBasis(4), 2)
nlevel3 = basisstate(NLevelBasis(3), 3)
@test sprint((io, x) -> show(IOContext(io, :limit=>true, :displaysize=>(16,80)), x), (nlevel1 ⊗ nlevel2) ⊗ dagger(nlevel3)) ==
"Operator(dim=16x3)\n  basis left:  [NLevel(N=4) ⊗ NLevel(N=4)]\n  basis right: NLevel(N=3)\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  1.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n    ⋮                           \n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im"

# Test switching back
QuantumOpticsBase.set_printing(standard_order=false)
state_data = kron(spin2.data, spin1.data, n.data)
state_data_str = sprint(Base.print_array, state_data)
@test sprint(show, state) == "Ket(dim=12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n"*state_data_str


end # testset
