using Base.Test
using QuantumOptics

type TestOperator <: Operator
end

@testset "sparseoperator" begin

srand(0)

# Set up operators
spinbasis = SpinBasis(1//2)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)

sx_dense = full(sx)
sy_dense = full(sy)
sz_dense = full(sz)

@test typeof(sx_dense) == DenseOperator
@test typeof(sparse(sx_dense)) == SparseOperator
@test sparse(sx_dense) == sx

b = FockBasis(3)
I = identityoperator(b)


a = TestOperator()

@test_throws ArgumentError sparse(a)

@test diagonaloperator(b, [1, 1, 1, 1]) == I
@test diagonaloperator(b, [1., 1., 1., 1.]) == I
@test diagonaloperator(b, [1im, 1im, 1im, 1im]) == 1im*I
@test diagonaloperator(b, [0:3;]) == number(b)

end # testset
