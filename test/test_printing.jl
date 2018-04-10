using Base.Test
using QuantumOptics

@testset "printing" begin

@test sprint(show, GenericBasis([2, 3])) == "Basis(shape=[2,3])"
@test sprint(show, GenericBasis(2)) == "Basis(dim=2)"
@test sprint(show, SpinBasis(1//1)) == "Spin(1)"
@test sprint(show, SpinBasis(3//2)) == "Spin(3/2)"
@test sprint(show, FockBasis(1)) == "Fock(cutoff=1)"
@test sprint(show, NLevelBasis(2)) == "NLevel(N=2)"

@test sprint(show, PositionBasis(-4, 4, 10)) == "Position(xmin=-4.0, xmax=4.0, N=10)"
@test sprint(show, MomentumBasis(-4, 4, 10)) == "Momentum(pmin=-4.0, pmax=4.0, N=10)"

b_fock = FockBasis(4)
states = [fockstate(b_fock, 2), coherentstate(b_fock, 0.4)]
@test sprint(show, SubspaceBasis(states)) == "Subspace(superbasis=Fock(cutoff=4), states:2)"

b_mb = ManyBodyBasis(b_fock, fermionstates(b_fock, 2))
@test sprint(show, b_mb) == "ManyBody(onebodybasis=Fock(cutoff=4), states:10)"

state = fockstate(FockBasis(2), 2)
@test sprint(show, state) == "Ket(dim=3)\n  basis: Fock(cutoff=2)\n 0.0+0.0im\n 0.0+0.0im\n 1.0+0.0im"
state = dagger(state)
@test sprint(show, state) == "Bra(dim=3)\n  basis: Fock(cutoff=2)\n 0.0-0.0im\n 0.0-0.0im\n 1.0-0.0im"

op = DenseOperator(FockBasis(1))
@test sprint(show, op) == "DenseOperator(dim=2x2)
  basis: Fock(cutoff=1)
 0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im"

op = DenseOperator(b_fock, b_fock ⊗ SpinBasis(1//2))
@test sprint(show, op) == "DenseOperator(dim=5x10)
  basis left:  Fock(cutoff=4)
  basis right: [Fock(cutoff=4) ⊗ Spin(1/2)]
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im"

op = SparseOperator(b_fock, b_fock ⊗ SpinBasis(1//2))
@test sprint(show, op) == "SparseOperator(dim=5x10)
  basis left:  Fock(cutoff=4)
  basis right: [Fock(cutoff=4) ⊗ Spin(1/2)]
    []"

op = SparseOperator(b_fock)
op.data[2,2] = 1
@test replace(sprint(show, op), "\t", "  ") == "SparseOperator(dim=5x5)
  basis: Fock(cutoff=4)\n  [2, 2]  =  1.0+0.0im"

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
  basis right: Position(xmin=-2.0, xmax=2.0, N=4)"

# Inversed tensor product ordering
QuantumOptics.set_printing(standard_order=true)

n = fockstate(b_fock, 1)
@test sprint(show, n) == "Ket(dim=3)\n  basis: Fock(cutoff=2)\n 0.0+0.0im\n 1.0+0.0im\n 0.0+0.0im"

spin1 = spindown(b_spin)
spin2 = spinup(b_spin)
state = n ⊗ spin1 ⊗ spin2
state_data = kron(n.data, spin1.data, spin2.data)
type_len = length("Complex{Float64}")
state_data_str = join(split(sprint(show, state_data)[type_len+2:end-1], ','), "\n")
@test sprint(show, state) == "Ket(dim=12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n "*state_data_str

state_data_str = join(split(sprint(show, state_data')[type_len+2:end-1]), "\n ")
@test sprint(show, dagger(state)) == "Bra(dim=12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n "*state_data_str

op = dm(state)
op_data = state_data * state_data'
op_data_str1 = split(sprint(show, op_data)[type_len+2:end-1], ";")
for i=1:length(op_data_str1)
    op_data_str1[i] = join(split(op_data_str1[i]), "  ")
end
op_data_str = join(op_data_str1, "\n ")
@test sprint(show, op) == "DenseOperator(dim=12x12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n "*op_data_str

op = sparse(op)
op_data = sparse(op_data)
op_data_str = sprint(show, op_data)[4:end]
@test sprint(show, op) == "SparseOperator(dim=12x12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n  "*op_data_str

# Test switching back
QuantumOptics.set_printing(standard_order=false)
state_data = kron(spin2.data, spin1.data, n.data)
state_data_str = join(split(sprint(show, state_data)[type_len+2:end-1], ','), "\n")
@test sprint(show, state) == "Ket(dim=12)
  basis: [Fock(cutoff=2) ⊗ Spin(1/2) ⊗ Spin(1/2)]\n "*state_data_str


end # testset
