module printing

export set_printing

import Base: show

using Compat
using ..bases, ..states
using ..operators, ..operators_dense, ..operators_sparse
using ..operators_lazytensor, ..operators_lazysum, ..operators_lazyproduct
using ..spin, ..fock, ..nlevel, ..particle, ..subspace, ..manybody, ..sparsematrix


"""
    QuantumOptics.set_printing(; standard_order, rounding_tol)

Set options for REPL output.

# Arguments
* `standard_order=false`: For performance reasons, the order of the tensor
    product is inverted, i.e. `tensor(a, b)=kron(b, a)`. When changing this
    to `true`, the output shown in the REPL will exhibit the correct order.
* `rounding_tol=1e-17`: Tolerance for floating point errors shown in the output.
"""
function set_printing(; standard_order::Bool=_std_order, rounding_tol::Real=_round_tol)
    global _std_order = standard_order
    global _round_tol = rounding_tol
    global machineprecorder = Int32(round(-log10(_round_tol), 0))
    nothing
end
set_printing(standard_order=false, rounding_tol=1e-17)

function show(stream::IO, x::GenericBasis)
    if length(x.shape) == 1
        write(stream, "Basis(dim=$(x.shape[1]))")
    else
        s = replace(string(x.shape), " ", "")
        write(stream, "Basis(shape=$s)")
    end
end

function show(stream::IO, x::CompositeBasis)
    write(stream, "[")
    for i in 1:length(x.bases)
        show(stream, x.bases[i])
        if i != length(x.bases)
            write(stream, " ⊗ ")
        end
    end
    write(stream, "]")
end

function show(stream::IO, x::SpinBasis)
    d = denominator(x.spinnumber)
    n = numerator(x.spinnumber)
    if d == 1
        write(stream, "Spin($n)")
    else
        write(stream, "Spin($n/$d)")
    end
end

function show(stream::IO, x::FockBasis)
    write(stream, "Fock(cutoff=$(x.N))")
end

function show(stream::IO, x::NLevelBasis)
    write(stream, "NLevel(N=$(x.N))")
end

function show(stream::IO, x::PositionBasis)
    write(stream, "Position(xmin=$(x.xmin), xmax=$(x.xmax), N=$(x.N))")
end

function show(stream::IO, x::MomentumBasis)
    write(stream, "Momentum(pmin=$(x.pmin), pmax=$(x.pmax), N=$(x.N))")
end

function show(stream::IO, x::SubspaceBasis)
    write(stream, "Subspace(superbasis=$(x.superbasis), states:$(length(x.basisstates)))")
end

function show(stream::IO, x::ManyBodyBasis)
    write(stream, "ManyBody(onebodybasis=$(x.onebodybasis), states:$(length(x.occupations)))")
end

function show(stream::IO, x::Ket)
    write(stream, "Ket(dim=$(length(x.basis)))\n  basis: $(x.basis)\n")
    if !_std_order
        Base.showarray(stream, x.data, false; header=false)
    else
        showquantumstatebody(stream, x)
    end
end

function show(stream::IO, x::Bra)
    write(stream, "Bra(dim=$(length(x.basis)))\n  basis: $(x.basis)\n")
    if !_std_order
        Base.showarray(stream, x.data, false; header=false)
    else
        showquantumstatebody(stream, x)
    end
end

function showoperatorheader(stream::IO, x::Operator)
    write(stream, "$(typeof(x).name.name)(dim=$(length(x.basis_l))x$(length(x.basis_r)))\n")
    if bases.samebases(x)
        write(stream, "  basis: ")
        show(stream, basis(x))
    else
        write(stream, "  basis left:  ")
        show(stream, x.basis_l)
        write(stream, "\n  basis right: ")
        show(stream, x.basis_r)
    end
end

show(stream::IO, x::Operator) = showoperatorheader(stream, x)

function showquantumstatebody(stream::IO, x::Union{Ket,Bra})
    #the permutation is used to invert the order A x B = B.data x A.data to A.data x B.data
    perm = collect(length(basis(x).shape):-1:1)
    if length(perm) == 1
        Base.showarray(stream, round.(x.data, machineprecorder), false; header=false)
    else
        Base.showarray(stream,
        round.(permutesystems(x,perm).data, machineprecorder), false; header=false)
    end
end

function permuted_densedata(x::DenseOperator)
    lbn = length(x.basis_l.shape)
    rbn = length(x.basis_r.shape)
    perm = collect(max(lbn,rbn):-1:1)
    #padd the shape with additional x1 subsystems s.t. x has symmetric number of subsystems
    decomp = lbn > rbn ? [x.basis_l.shape; x.basis_r.shape; fill(1,lbn-rbn)] :
                         [x.basis_l.shape; fill(1,rbn-lbn); x.basis_r.shape]

    data = reshape(x.data, decomp...)
    data = permutedims(data, [perm; perm + length(perm)])
    data = reshape(data, length(x.basis_l), length(x.basis_r))

    return round.(data, machineprecorder)
end

function permuted_sparsedata(x::SparseOperator)
    lbn = length(x.basis_l.shape)
    rbn = length(x.basis_r.shape)
    perm = collect(max(lbn,rbn):-1:1)
    #padd the shape with additional x1 subsystems s.t. x has symmetric number of subsystems
    decomp = lbn > rbn ? [x.basis_l.shape; x.basis_r.shape; fill(1,lbn-rbn)] :
                         [x.basis_l.shape; fill(1,rbn-lbn); x.basis_r.shape]

    data = sparsematrix.permutedims(x.data, decomp, [perm; perm + length(perm)])

    return round.(data, machineprecorder)
end


function show(stream::IO, x::DenseOperator)
    showoperatorheader(stream, x)
    write(stream, "\n")
    if !_std_order
        Base.showarray(stream, x.data, false; header=false)
    else
        Base.showarray(stream, permuted_densedata(x), false; header=false)
    end
end

function show(stream::IO, x::SparseOperator)
    showoperatorheader(stream, x)
    if nnz(x.data) == 0
        write(stream, "\n    []")
    else
        if !_std_order
            show(stream, x.data)
        else
            show(stream, permuted_sparsedata(x))
        end
    end
end

function show(stream::IO, x::LazyTensor)
    showoperatorheader(stream, x)
    write(stream, "\n  operators: $(length(x.operators))")
    s = replace(string(x.indices), " ", "")
    write(stream, "\n  indices: $s")
end

function show(stream::IO, x::Union{LazySum, LazyProduct})
    showoperatorheader(stream, x)
    write(stream, "\n  operators: $(length(x.operators))")
end


end # module
