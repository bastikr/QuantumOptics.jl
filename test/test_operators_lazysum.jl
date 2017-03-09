using Base.Test
using QuantumOptics


@testset "operators-lazysum" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)
randop(bl, br) = DenseOperator(bl, br, rand(Complex128, length(bl), length(br)))
randop(b) = randop(b, b)
sprandop(bl, br) = sparse(DenseOperator(bl, br, rand(Complex128, length(bl), length(br))))
sprandop(b) = sprandop(b, b)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

# Test creation
@test_throws AssertionError LazySum()
@test_throws AssertionError LazySum([1., 2.], [randop(b_l)])
@test_throws AssertionError LazySum(randop(b_l, b_r), sparse(randop(b_l, b_l)))
@test_throws AssertionError LazySum(randop(b_l, b_r), sparse(randop(b_r, b_r)))

# Test full & sparse
op1 = randop(b_l, b_r)
op2 = sparse(randop(b_l, b_r))
@test 0.1*op1 + 0.3*full(op2) == full(LazySum([0.1, 0.3], [op1, op2]))
@test 0.1*sparse(op1) + 0.3*op2 == sparse(LazySum([0.1, 0.3], [op1, op2]))


# Arithmetic operations
# =====================
op1a = randop(b_l, b_r)
op1b = randop(b_l, b_r)
op2a = randop(b_l, b_r)
op2b = randop(b_l, b_r)
op3a = randop(b_l, b_r)
op1 = LazySum([0.1, 0.3], [op1a, sparse(op1b)])
op1_ = 0.1*op1a + 0.3*op1b
op2 = LazySum([0.7, 0.9], [sparse(op2a), op2b])
op2_ = 0.7*op2a + 0.9*op2b
op3 = LazySum(op3a)
op3_ = op3a

x1 = Ket(b_r, rand(Complex128, length(b_r)))
x2 = Ket(b_r, rand(Complex128, length(b_r)))
xbra1 = Bra(b_l, rand(Complex128, length(b_l)))

# Addition
@test_throws bases.IncompatibleBases op1 + dagger(op2)
@test 1e-14 > D(op1+op2, op1_+op2_)

# Subtraction
@test_throws bases.IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1 - op2, op1_ - op2_)
@test 1e-14 > D(op1 + (-op2), op1_ - op2_)
@test 1e-14 > D(op1 + (-1*op2), op1_ - op2_)

# Test multiplication
@test_throws ArgumentError op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), (op1_+op2_)*(x1+0.3*x2))
@test 1e-12 > D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))

# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Test identityoperator
Idense = identityoperator(DenseOperator, b_r)
I = identityoperator(LazySum, b_r)
@test isa(I, LazySum)
@test full(I) == Idense
@test 1e-11 > D(I*x1, x1)

Idense = identityoperator(DenseOperator, b_l)
I = identityoperator(LazySum, b_l)
@test isa(I, LazySum)
@test full(I) == Idense
@test 1e-11 > D(xbra1*I, xbra1)

# Test trace and normalize
op1 = randop(b_l)
op2 = randop(b_l)
op3 = randop(b_l)
op = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

@test trace(op_) ≈ trace(op)
op_normalized = normalize(op)
@test trace(op_) ≈ trace(op)
@test 1 ≈ trace(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test trace(op) != trace(op_copy)
@test 1 ≈ trace(op_copy)

# Test partial trace
op1 = randop(b_l)
op2 = randop(b_l)
op3 = randop(b_l)
op123 = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op123_ = 0.1*op1 + 0.3*op2 + 1.2*op3

@test 1e-14 > D(ptrace(op123_, 3), ptrace(op123, 3))
@test 1e-14 > D(ptrace(op123_, 2), ptrace(op123, 2))
@test 1e-14 > D(ptrace(op123_, 1), ptrace(op123, 1))

@test 1e-14 > D(ptrace(op123_, [2,3]), ptrace(op123, [2,3]))
@test 1e-14 > D(ptrace(op123_, [1,3]), ptrace(op123, [1,3]))
@test 1e-14 > D(ptrace(op123_, [1,2]), ptrace(op123, [1,2]))

@test 1e-14 > abs(ptrace(op123_, [1,2,3]) - ptrace(op123, [1,2,3]))

# Test expect
state = Ket(b_l, rand(Complex128, length(b_l)))
@test expect(op123, state) ≈ expect(op123_, state)

state = DenseOperator(b_l, b_l, rand(Complex128, length(b_l), length(b_l)))
@test expect(op123, state) ≈ expect(op123_, state)

# Permute systems
op1a = randop(b1a)
op2a = randop(b2a)
op3a = randop(b3a)
op1b = randop(b1a)
op2b = randop(b2a)
op3b = randop(b3a)
op1c = randop(b1a)
op2c = randop(b2a)
op3c = randop(b3a)
op123a = op1a⊗op2a⊗op3a
op123b = op1b⊗op2b⊗op3b
op123c = op1c⊗op2c⊗op3c
op = LazySum([0.3, 0.7, 1.2], [op123a, sparse(op123b), op123c])
op_ = 0.3*op123a + 0.7*op123b + 1.2*op123c

@test 1e-14 > D(permutesystems(op, [1, 3, 2]), permutesystems(op_, [1, 3, 2]))
@test 1e-14 > D(permutesystems(op, [2, 1, 3]), permutesystems(op_, [2, 1, 3]))
@test 1e-14 > D(permutesystems(op, [2, 3, 1]), permutesystems(op_, [2, 3, 1]))
@test 1e-14 > D(permutesystems(op, [3, 1, 2]), permutesystems(op_, [3, 1, 2]))
@test 1e-14 > D(permutesystems(op, [3, 2, 1]), permutesystems(op_, [3, 2, 1]))


# Test gemv
op1 = randop(b_l, b_r)
op2 = randop(b_l, b_r)
op3 = randop(b_l, b_r)
op = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

state = Ket(b_r, rand(Complex128, length(b_r)))
result_ = Ket(b_l, rand(Complex128, length(b_l)))
result = deepcopy(result_)
operators.gemv!(complex(1.), op, state, complex(0.), result)
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, op, state, beta, result)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

state = Bra(b_l, rand(Complex128, length(b_l)))
result_ = Bra(b_r, rand(Complex128, length(b_r)))
result = deepcopy(result_)
operators.gemv!(complex(1.), state, op, complex(0.), result)
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, state, op, beta, result)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm
op1 = randop(b_l, b_r)
op2 = randop(b_l, b_r)
op3 = randop(b_l, b_r)
op = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

state = randop(b_r, b_r)
result_ = randop(b_l, b_r)
result = deepcopy(result_)
operators.gemm!(complex(1.), op, state, complex(0.), result)
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, op, state, beta, result)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randop(b_l, b_l)
result_ = randop(b_l, b_r)
result = deepcopy(result_)
operators.gemm!(complex(1.), state, op, complex(0.), result)
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, state, op, beta, result)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

end # testset
