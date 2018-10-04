module semiclassical

import Base: ==
import ..bases, ..operators, ..operators_dense
import ..timeevolution: integrate, recast!

using ..bases, ..states, ..operators, ..operators_dense, ..timeevolution

import LinearAlgebra: mul!

const QuantumState{B<:Basis} = Union{Ket{B}, Operator{B,B}}
const DecayRates = Union{Nothing, Vector{Float64}, Matrix{Float64}}

"""
Semi-classical state.

It consists of a quantum part, which is either a `Ket` or a `Operator` and
a classical part that is specified as a complex vector of arbitrary length.
"""
mutable struct State{T<:QuantumState{Basis},C<:Vector{ComplexF64}}
    quantum::T
    classical::C
end

Base.length(state::State) = length(state.quantum) + length(state.classical)
Base.copy(state::State) = State(copy(state.quantum), copy(state.classical))

function ==(a::State{T,C}, b::State{T,C}) where {T<:QuantumState,C<:Vector{ComplexF64}}
    length(a.classical)==length(b.classical) &&
    (a.classical==b.classical) &&
    (a.quantum==b.quantum)
end

operators.expect(op, state::State) = expect(op, state.quantum)
operators.variance(op, state::State) = variance(op, state.quantum)
operators.ptrace(state::State, indices::Vector{Int}) = State{Operator}(ptrace(state.quantum, indices), state.classical)

operators_dense.dm(x::State{T}) where T<:Ket = State{Operator}(dm(x.quantum), x.classical)


"""
    semiclassical.schroedinger_dynamic(tspan, state0, fquantum, fclassical[; fout, ...])

Integrate time-dependent Schrödinger equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, psi, u) -> H` returning the time and or state
        dependent Hamiltonian.
* `fclassical`: Function `f(t, psi, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_dynamic(tspan, state0::State{T}, fquantum::Function, fclassical::Function;
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where T <: Ket
    tspan_ = convert(Vector{Float64}, tspan)
    dschroedinger_(t, state::State{T}, dstate::State{T}) = dschroedinger_dynamic(t, state, fquantum, fclassical, dstate)
    x0 = Vector{ComplexF64}(undef, length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    integrate(tspan_, dschroedinger_, x0, state, dstate, fout; kwargs...)
end

"""
    semiclassical.master_dynamic(tspan, state0, fquantum, fclassical; <keyword arguments>)

Integrate time-dependent master equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, rho, u) -> (H, J, Jdagger)` returning the time
        and/or state dependent Hamiltonian and Jump operators.
* `fclassical`: Function `f(t, rho, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the complex vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is not
        permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan, state0::State{T}, fquantum, fclassical;
                rates::Union{Vector{Float64}, Matrix{Float64}, Nothing}=nothing,
                fout::Union{Function,Nothing}=nothing,
                tmp::T=copy(state0.quantum),
                kwargs...) where T<:Operator
    tspan_ = convert(Vector{Float64}, tspan)
    function dmaster_(t, state::State{T}, dstate::State{T})
        dmaster_h_dynamic(t, state, fquantum, fclassical, rates, dstate, tmp)
    end
    x0 = Vector{ComplexF64}(undef, length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    integrate(tspan_, dmaster_, x0, state, dstate, fout; kwargs...)
end

function master_dynamic(tspan, state0::State{T}, fquantum, fclassical; kwargs...) where T<:Ket
    master_dynamic(tspan, dm(state0), fquantum, fclassical; kwargs...)
end

# TODO: recast! for sparse operators
function recast!(state::State, x::Vector{ComplexF64})
    N = length(state.quantum)
    copyto!(x, 1, state.quantum.data, 1, N)
    copyto!(x, N+1, state.classical, 1, length(state.classical))
    x
end

function recast!(x::Vector{ComplexF64}, state::State)
    N = length(state.quantum)
    copyto!(state.quantum.data, 1, x, 1, N)
    copyto!(state.classical, 1, x, N+1, length(state.classical))
end

function dschroedinger_dynamic(t::Float64, state::State{T,C}, fquantum::Function,
            fclassical::Function, dstate::State{T,C}) where {T<:Ket,C<:Vector{ComplexF64}}
    fquantum_(t, psi) = fquantum(t, state.quantum, state.classical)
    timeevolution.timeevolution_schroedinger.dschroedinger_dynamic(t, state.quantum, fquantum_, dstate.quantum)
    fclassical(t, state.quantum, state.classical, dstate.classical)
end

function dmaster_h_dynamic(t::Float64, state::State{T,C}, fquantum::Function,
            fclassical::Function, rates::DecayRates, dstate::State{T,C}, tmp::T) where {T<:Operator,C<:Vector{ComplexF64}}
    fquantum_(t, rho) = fquantum(t, state.quantum, state.classical)
    timeevolution.timeevolution_master.dmaster_h_dynamic(t, state.quantum, fquantum_, rates, dstate.quantum, tmp)
    fclassical(t, state.quantum, state.classical, dstate.classical)
end

end # module
