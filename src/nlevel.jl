module nlevel

import Base.==

using ..bases, ..states, ..operators, ..operators_sparse

export NLevelBasis, transition


type NLevelBasis <: Basis
    shape::Vector{Int}
    N::Int
    function NLevelBasis(N::Int)
        if N < 1
            throw(DimensionMismatch())
        end
        new([N], N)
    end
end

==(b1::NLevelBasis, b2::NLevelBasis) = b1.N == b2.N

function transition(b::NLevelBasis, to::Int, from::Int)
    op = SparseOperator(b)
    op.data[to, from] = 1.
    op
end


end # module