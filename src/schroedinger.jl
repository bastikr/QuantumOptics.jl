"""
    timeevolution.schroedinger(tspan, psi0, H; fout)

Integrate Schroedinger equation to evolve states or compute propagators.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `psi0`: Initial state vector (can be a bra or a ket) or an Operator from some basis to the basis of the Hamiltonian (psi0.basis_l == basis(H)).
* `H`: Arbitrary operator specifying the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, psi)` is called every time
        an output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver and
        therefore must not be changed.
"""
function schroedinger(tspan, psi0::T, H::AbstractOperator{B,B};
                fout=nothing,
                kwargs...) where {B,Bo,T<:Union{AbstractOperator{B,Bo},StateVector{B}}}
    _check_const(H)
    dschroedinger_(t, psi, dpsi) = dschroedinger!(dpsi, H, psi)
    tspan, psi0 = _promote_time_and_state(psi0, H, tspan) # promote only if ForwardDiff.Dual
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate(tspan, dschroedinger_, x0, state, dstate, fout; kwargs...)
end


"""
    timeevolution.schroedinger_dynamic(tspan, psi0, f; fout)

Integrate time-dependent Schroedinger equation to evolve states or compute propagators.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `psi0`: Initial state vector (can be a bra or a ket) or an Operator from some basis to the basis of the Hamiltonian (psi0.basis_l == basis(H)).
* `f`: Function `f(t, psi) -> H` returning the time and or state dependent Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, psi)` is called every time
        an output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver and
        therefore must not be changed.

    timeevolution.schroedinger_dynamic(tspan, psi0, H::AbstractTimeDependentOperator; fout)

Instead of a function `f`, this takes a time-dependent operator `H`.
"""
function schroedinger_dynamic(tspan, psi0, f;
                fout=nothing,
                kwargs...)
    dschroedinger_ = let f = f
        dschroedinger_(t, psi, dpsi) = dschroedinger_dynamic!(dpsi, f, psi, t)
    end
    tspan, psi0 = _promote_time_and_state(psi0, f, tspan) # promote only if ForwardDiff.Dual
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate(tspan, dschroedinger_, x0, state, dstate, fout; kwargs...)
end

function schroedinger_dynamic(tspan, psi0::T, H::AbstractTimeDependentOperator;
    kwargs...) where {B,Bp,T<:Union{AbstractOperator{B,Bp},StateVector{B}}}
    promoted_tspan, psi0 = _promote_time_and_state(psi0, H, tspan)
    if promoted_tspan !== tspan # promote H
        promoted_H = TimeDependentSum(H.coefficients, H.static_op.operators; init_time=first(promoted_tspan))
        return schroedinger_dynamic(promoted_tspan, psi0, schroedinger_dynamic_function(promoted_H); kwargs...)
    else
        return schroedinger_dynamic(promoted_tspan, psi0, schroedinger_dynamic_function(H); kwargs...)
    end
end

"""
    recast!(x,y)

Write the data stored in `y` into `x`, where either `x` or `y` is a quantum
object such as a [`Ket`](@ref) or an [`Operator`](@ref), and the other one is
a vector or a matrix with a matching size.
"""
recast!(psi::StateVector{B,D},x::D) where {B, D} = (psi.data = x);
recast!(x::D,psi::StateVector{B,D}) where {B, D} = nothing
function recast!(proj::Operator{B1,B2,T},x::T) where {B1,B2,T}
    proj.data = x
end
recast!(x::T,proj::Operator{B1,B2,T}) where {B1,B2,T} = nothing

"""
    dschroedinger!(dpsi, H, psi)

Update the increment `dpsi` in-place according to a Schrödinger equation
as `-im*H*psi`.

See also: [`dschroedinger_dynamic!`](@ref)
"""
function dschroedinger!(dpsi, H, psi)
    QuantumOpticsBase.mul!(dpsi,H,psi,eltype(psi)(-im),zero(eltype(psi)))
    return dpsi
end

function dschroedinger!(dpsi, H, psi::Bra)
    QuantumOpticsBase.mul!(dpsi,psi,H,eltype(psi)(im),zero(eltype(psi)))
    return dpsi
end

"""
    dschroedinger_dynamic!(dpsi, f, psi, t)

Compute the Hamiltonian as `H=f(t, psi)` and update `dpsi` according to a
Schrödinger equation as `-im*H*psi`.

See also: [`dschroedinger!`](@ref)
"""
function dschroedinger_dynamic!(dpsi, f::F, psi, t) where {F}
    H = f(t, psi)
    dschroedinger!(dpsi, H, psi)
end


function check_schroedinger(psi, H)
    check_multiplicable(H, psi)
    check_samebases(H)
end

function check_schroedinger(psi::Bra, H)
    check_multiplicable(psi, H)
    check_samebases(H)
end


function _promote_time_and_state(u0, H::AbstractOperator, tspan)
    Ts = eltype(H)
    Tt = real(Ts)
    p = Vector{Tt}(undef,0)
    u0_promote = DiffEqBase.promote_u0(u0, p, tspan[1])
    tspan_promote = DiffEqBase.promote_tspan(u0_promote, p, tspan, nothing, Dict{Symbol, Any}())
    return tspan_promote, u0_promote
end
_promote_time_and_state(u0, f, tspan) = _promote_time_and_state(u0, f(first(tspan), u0), tspan)

@inline function DiffEqBase.promote_u0(u0::T, p, t0) where {T<:Union{Bra,Ket}}
    u0data_promote = DiffEqBase.promote_u0(u0.data, p, t0)
    if u0data_promote !== u0.data
        u0_promote = T(u0.basis, u0data_promote)
        return u0_promote
    end
    return u0
end
@inline function DiffEqBase.promote_u0(u0::Operator, p, t0)
    u0data_promote = DiffEqBase.promote_u0(u0.data, p, t0)
    if u0data_promote !== u0.data
        u0_promote = Operator(u0.basis_l, u0.basis_r, u0data_promote)
        return u0_promote
    end
    return u0
end
