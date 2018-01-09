using ..ode_dopri, ..metrics

import OrdinaryDiffEq, DiffEqCallbacks

function recast! end

"""
df(t, state::T, dstate::T)
"""
function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, fout::Function;
            steady_state = false, eps = 1e-3, kwargs...)

    function df_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        recast!(x, state)
        recast!(dx, dstate)
        df(t, state, dstate)
        recast!(dstate, dx)
    end
    function fout_(t::Float64, x::Vector{Complex128},integrator)
        recast!(x, state)
        fout(t, state)
    end

    # TODO: Infer the output of `fout` instead of computing it
    recast!(x0, state)
    out = DiffEqCallbacks.SavedValues(Float64,typeof(fout(tspan[1], state)))
    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=tspan)

    # Build callback solve with DP5
    # TODO: Expose algorithm choice

    prob = OrdinaryDiffEq.ODEProblem{true}(df_, x0,(tspan[1],tspan[end]))

    if steady_state
        _cb = OrdinaryDiffEq.DiscreteCallback(
                                SteadyStateCondtion(copy(state),eps,state),
                                (integrator)->OrdinaryDiffEq.terminate!(integrator),
                                save_positions = (false,false))
        cb = OrdinaryDiffEq.CallbackSet(_cb,scb)
        sol = OrdinaryDiffEq.solve(
                    prob,
                    OrdinaryDiffEq.DP5();
                    reltol = 1.0e-6,
                    abstol = 1.0e-8,
                    save_everystep = false, save_start = false,
                    callback=cb, kwargs...)
        # TODO: On v0.7 it's type-stable to return only sol.u[end]!
        return sol.t,sol.u
    else
        sol = OrdinaryDiffEq.solve(
                    prob,
                    OrdinaryDiffEq.DP5();
                    reltol = 1.0e-6,
                    abstol = 1.0e-8,
                    save_everystep = false, save_start = false, save_end = false,
                    callback=scb, kwargs...)
        return out.t,out.saveval
    end
end

function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, ::Void; kwargs...)
    function fout(t::Float64, state::T)
        copy(state)
    end
    integrate(tspan, df, x0, state, dstate, fout; kwargs...)
end

struct SteadyStateCondtion{T,T2,T3}
    rho0::T
    eps::T2
    state::T3
end
function (c::SteadyStateCondtion)(t,rho,integrator)
    timeevolution.recast!(rho,c.state)
    dt = integrator.dt
    drho = metrics.tracedistance(c.rho0, c.state)
    c.rho0.data[:] = c.state.data
    drho/dt < c.eps
end
