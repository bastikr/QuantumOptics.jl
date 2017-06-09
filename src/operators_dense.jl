module operators_dense

import Base: trace, ==, +, -, *, /, ishermitian
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems,
                    gemv!, gemm!

using Base.LinAlg
using Base.Cartesian
using ..bases, ..states, ..operators

export DenseOperator, projector, dm

"""
    DenseOperator(b1[, b2, data])

Dense array implementation of Operator.

The matrix consisting of complex floats is stored in the `data` field.
"""
type DenseOperator <: Operator
    basis_l::Basis
    basis_r::Basis
    data::Matrix{Complex128}
    DenseOperator(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new(b1, b2, data) : throw(DimensionMismatch())
end

DenseOperator(b::Basis, data) = DenseOperator(b, b, data)
DenseOperator(b1::Basis, b2::Basis) = DenseOperator(b1, b2, zeros(Complex128, length(b1), length(b2)))
DenseOperator(b::Basis) = DenseOperator(b, b)
DenseOperator(op::Operator) = full(op)

Base.copy(x::DenseOperator) = DenseOperator(x.basis_l, x.basis_r, copy(x.data))

"""
    full(op::Operator)

Convert an arbitrary Operator into a [`DenseOperator`](@ref).
"""
Base.full(x::Operator) = throw(ArgumentError("Conversion from $(typeof(a)) to a DenseOperator not implemented."))
Base.full(x::DenseOperator) = copy(x)

==(x::DenseOperator, y::DenseOperator) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && (x.data == y.data)


# Arithmetic operations
*(a::DenseOperator, b::Ket) = (check_multiplicable(a, b); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::DenseOperator) = (check_multiplicable(a, b); Bra(b.basis_r, b.data.'*a.data))
*(a::DenseOperator, b::DenseOperator) = (check_multiplicable(a, b); DenseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::DenseOperator, b::Number) = DenseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::DenseOperator) = DenseOperator(b.basis_l, b.basis_r, complex(a)*b.data)
function *(op1::Operator, op2::DenseOperator)
    check_multiplicable(op1, op2)
    result = DenseOperator(op1.basis_l, op2.basis_r)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end

function *(op1::DenseOperator, op2::Operator)
    check_multiplicable(op1, op2)
    result = DenseOperator(op1.basis_l, op2.basis_r)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end

function *(op::Operator, psi::Ket)
    check_multiplicable(op, psi)
    result = Ket(op.basis_l)
    gemv!(Complex(1.), op, psi, Complex(0.), result)
    return result
end

function *(psi::Bra, op::Operator)
    check_multiplicable(psi, op)
    result = Bra(op.basis_r)
    gemv!(Complex(1.), psi, op, Complex(0.), result)
    return result
end

/(a::DenseOperator, b::Number) = DenseOperator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::DenseOperator, b::DenseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::DenseOperator) = DenseOperator(a.basis_l, a.basis_r, -a.data)
-(a::DenseOperator, b::DenseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data-b.data))

dagger(x::DenseOperator) = DenseOperator(x.basis_r, x.basis_l, x.data')

identityoperator(::Type{DenseOperator}, b1::Basis, b2::Basis) = DenseOperator(b1, b2, eye(Complex128, length(b1), length(b2)))

trace(op::DenseOperator) = (check_samebases(op); trace(op.data))

function ptrace(a::DenseOperator, indices::Vector{Int})
    operators.check_ptrace_arguments(a, indices)
    if length(a.basis_l.shape) == length(indices)
        return trace(a)
    end
    rank = length(a.basis_l.shape)
    result = _ptrace(Val{rank}(), a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return DenseOperator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end

ptrace(a::Ket, indices::Vector{Int}) = ptrace(tensor(a, dagger(a)), indices)
ptrace(a::Bra, indices::Vector{Int}) = ptrace(tensor(dagger(a), a), indices)

states.normalize!(op::DenseOperator) = scale!(op.data, 1./trace(op))

function operators.expect(op::DenseOperator, state::Operator)
    check_samebases(op.basis_r, state.basis_l)
    check_samebases(op.basis_l, state.basis_r)
    result = Complex128(0.)
    @inbounds for i=1:size(op.data, 1), j=1:size(op.data,2)
        result += op.data[i,j]*state.data[j,i]
    end
    result
end

tensor(a::DenseOperator, b::DenseOperator) = DenseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))
"""
    tensor(x::Ket, y::Bra)

Outer product ``|x⟩⟨y|`` of the given states.
"""
tensor(a::Ket, b::Bra) = DenseOperator(a.basis, b.basis, reshape(kron(b.data, a.data), prod(a.basis.shape), prod(b.basis.shape)))

function permutesystems(a::DenseOperator, perm::Vector{Int})
    @assert length(a.basis_l.bases) == length(a.basis_r.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(a.data, [a.basis_l.shape; a.basis_r.shape]...)
    data = permutedims(data, [perm; perm + length(perm)])
    data = reshape(data, length(a.basis_l), length(a.basis_r))
    DenseOperator(permutesystems(a.basis_l, perm), permutesystems(a.basis_r, perm), data)
end

"""
    projector(a::Ket, b::Bra)

Projection operator ``|a⟩⟨b|``.
"""
projector(a::Ket, b::Bra) = tensor(a, b)
"""
    projector(a::Ket)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Ket) = tensor(a, dagger(a))
"""
    projector(a::Bra)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Bra) = tensor(dagger(a), a)

"""
    dm(a::StateVector)

Create density matrix ``|a⟩⟨a|``. Same as `projector(a)`.
"""
dm(x::Ket) = tensor(x, dagger(x))
dm(x::Bra) = tensor(dagger(x), x)

function Base.expm(op::DenseOperator)
    check_samebases(op)
    return DenseOperator(op.basis_l, op.basis_r, expm(op.data))
end


# Partial trace for dense operators.
function _strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[1] = 1
    for m=2:N
        S[m] = S[m-1]*shape[m-1]
    end
    return S
end

@generated function _ptrace{RANK}(::Val{RANK}, a::Matrix{Complex128},
                                  shape_l::Vector{Int}, shape_r::Vector{Int},
                                  indices::Vector{Int})
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = copy(shape_l)
        result_shape_l[indices] = 1
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = copy(shape_r)
        result_shape_r[indices] = 1
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(Complex128, N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                #println("Jl_0: ", Jl_0, "; Jr_0: ", Jr_0, "; Il_0: ", Il_0, "; Ir_0: ", Ir_0)
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

# Fast in-place multiplication with dense operators
gemm!(alpha, a::Matrix{Complex128}, b::Matrix{Complex128}, beta, result::Matrix{Complex128}) = BLAS.gemm!('N', 'N', convert(Complex128, alpha), a, b, convert(Complex128, beta), result)
gemv!(alpha, a::Matrix{Complex128}, b::Vector{Complex128}, beta, result::Vector{Complex128}) = BLAS.gemv!('N', convert(Complex128, alpha), a, b, convert(Complex128, beta), result)
gemv!(alpha, a::Vector{Complex128}, b::Matrix{Complex128}, beta, result::Vector{Complex128}) = BLAS.gemv!('T', convert(Complex128, alpha), b, a, convert(Complex128, beta), result)

gemm!(alpha, a::DenseOperator, b::DenseOperator, beta, result::DenseOperator) = gemm!(convert(Complex128, alpha), a.data, b.data, convert(Complex128, beta), result.data)
gemv!(alpha, a::DenseOperator, b::Ket, beta, result::Ket) = gemv!(convert(Complex128, alpha), a.data, b.data, convert(Complex128, beta), result.data)
gemv!(alpha, a::Bra, b::DenseOperator, beta, result::Bra) = gemv!(convert(Complex128, alpha), a.data, b.data, convert(Complex128, beta), result.data)


# Multiplication for Operators in terms of their gemv! implementation
function gemm!(alpha, M::Operator, b::DenseOperator, beta, result::DenseOperator)
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        gemv!(alpha, M, bket, beta, resultket)
        result.data[:,i] = resultket.data
    end
end

function gemm!(alpha, b::DenseOperator, M::Operator, beta, result::DenseOperator)
    for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        gemv!(alpha, bbra, M, beta, resultbra)
        result.data[i,:] = resultbra.data
    end
end

ishermitian(A::DenseOperator) = ishermitian(A.data)

end # module
