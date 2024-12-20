"""
    bloch_redfield_tensor(H, a_ops; J=[], use_secular=true, secular_cutoff=0.1)

Create the super-operator for the Bloch-Redfield master equation such that ``\\dot ρ = R ρ`` based on the QuTiP implementation.

See QuTiP's documentation (http://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html) for more information and a brief derivation.


# Arguments
* `H`: Hamiltonian.
* `a_ops`: Nested list of [interaction operator, callback function] pairs for the Bloch-Redfield type processes where the callback function describes the environment spectrum for the corresponding interaction operator.
           The spectral functions must take the angular frequency as their only argument.
* `J=[]`: Vector containing the jump operators for the Linblad type processes (optional).
* `use_secular=true`: Specify whether or not to use the secular approximation.
* `secular_cutoff=0.1`: Cutoff to allow a degree of partial secularization. Terms are discarded if they are greater than (dw\\_min * secular cutoff) where dw\\_min is the smallest (non-zero) difference between any two eigenenergies of H.
                        This argument is only taken into account if use_secular=true.
"""
function bloch_redfield_tensor(H::AbstractOperator, a_ops; J=SparseOpType[], use_secular=true, secular_cutoff=0.1)
    _check_const(H)
    _check_const.(J)
    # Use the energy eigenbasis
    H_evals, transf_mat = eigen(Array(H.data)) #Array call makes sure H is a dense array
    H_ekets = [Ket(H.basis_l, transf_mat[:, i]) for i in 1:length(H_evals)]

    #Define function for transforming to Hamiltonian eigenbasis
    function to_Heb(op, U)
        oper = copy(op) #Aviod mutating input op
        oper.data = inv(U) * oper.data * U #Transform underlying array
        return oper
    end

    N = length(H_evals) #Hilbert space dimension
    K = length(a_ops) #Number of system-env interation operators

    # Calculate Liouvillian for Lindblad terms (unitary part + dissipation from J (if given)):
    Heb = to_Heb(H, transf_mat)
    #Use anon function
    f = (x->to_Heb(x, transf_mat))
    L_task = Threads.@spawn liouvillian(Heb, f.(J)) #This also includes unitary dynamics part dρ/dt = -i[H, ρ]
    # L = sparse(liouvillian(Heb, f.(J)))

    #If only Lindblad collapse terms (no a_ops given) then we're done
    if K==0
        L = sparse(fetch(L_task))
        return L, H_ekets #L is in the energy eigenbasis here
    end

    #Transform interaction operators to Hamiltonian eigenbasis
    A = Array{eltype(transf_mat)}(undef, N, N, K)
    for k in 1:K
        A[:, :, k] = to_Heb(a_ops[k][1], transf_mat).data
    end

    # Array of transition frequencies between eigenstates
    W = H_evals .- transpose(H_evals)

    #Array for spectral functions evaluated at transition frequencies
    Jw = Array{eltype(transf_mat)}(undef, N, N, K)
    # Jw = zeros(Complex{Float64}, N, N, K)
    for k in 1:K
        Jw[:, :, k] .= a_ops[k][2].(W)
    end

    #Calculate secular cutoff scale if needed
    if use_secular
        dw_min = minimum(abs.(W[W .!= 0.0]))
        w_cutoff = dw_min * secular_cutoff
    end

    #Initialize R_abcd array
    data = zeros(ComplexF64, N, N, N, N)
    #Loop through all indices and calculate elements - seems to be as efficient as any fancy broadcasting implementation (and much simpler to read)
    Threads.@threads for idx in CartesianIndices(data)

        a, b, c, d = Tuple(idx) #Unpack indices

        #Skip any values that are larger than the secular cutoff
        if use_secular && abs(W[a, b] - W[c, d]) > w_cutoff
            continue
        end

        """ Term 1 """
        sum!(view(data, idx), @views A[a, c, :] .* A[d, b, :] .* (Jw[c, a, :] .+ Jw[d, b, :]) ) #Broadcasting over interaction operators

        """ Term 2 (b == d) """
        if b == d
            data[idx] -= @views sum( A[a, :, :] .* A[:, c, :] .* Jw[c, :, :] ) #Broadcasting over interaction operators and extra sum over n
        end

        """ Term 3 (a == c) """
        if a == c
            data[idx] -= @views sum( A[d, :, :] .* A[:, b, :] .* Jw[d, :, :] ) #Broadcasting over interaction operators and extra sum over n
        end

    end

    data *= 0.5 #Don't forget the factor of 1/2
    data = reshape(data, N^2, N^2) #Convert to Liouville space
    R = sparse(data) #Remove any zero values and converts to sparse array

    #Add Bloch-Redfield part to unitary dyanmics and Lindblad Liouvillian calculated above
    L = sparse(fetch(L_task))
    L.data = L.data + R

    return L, H_ekets

end #Function


"""
    timeevolution.master_bloch_redfield(tspan, rho0, R, H; <keyword arguments>)

Time-evolution according to a Bloch-Redfield master equation.


# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `H`: Arbitrary operator specifying the Hamiltonian.
* `R`: Bloch-Redfield tensor describing the time-evolution ``\\dot ρ = R ρ`` (see timeevolution.bloch\\_redfield\\_tensor).
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_bloch_redfield(tspan,
        rho0::Operator{B,B}, L::SuperOperator{Tuple{B,B},Tuple{B,B}},
        H::BLROperator{B,B}; fout::Union{Function,Nothing}=nothing,
        kwargs...) where {B}

    #Prep basis transf
    evals, transf_mat = eigen(dense(H).data)
    transf_op = DenseOperator(rho0.basis_l, transf_mat)
    inv_transf_op = DenseOperator(rho0.basis_l, inv(transf_mat))

    # rho as Ket and L as DataOperator
    basis_comp = rho0.basis_l^2
    rho0_eb = Ket(basis_comp, (inv_transf_op * rho0 * transf_op).data[:]) #Transform to H eb and convert to Ket
    L_ = isa(L, SparseSuperOpType) ? SparseOperator(basis_comp, L.data) : DenseOperator(basis_comp, L.data)

    # Derivative function
    dmaster_br_(t, rho, drho) = dmaster_br(drho, rho, L_)

    return integrate_br(tspan, dmaster_br_, rho0_eb, transf_op, inv_transf_op, fout; kwargs...)
end
master_bloch_redfield(tspan, psi::Ket, args...; kwargs...) = master_bloch_redfield(tspan, dm(psi), args...; kwargs...)

# Derivative ∂ₜρ = Lρ
function dmaster_br(drho, rho, L)
    QuantumOpticsBase.mul!(drho,L,rho)
end

# Integrate if there is no fout specified
function integrate_br(tspan, dmaster_br, rho,
                transf_op, inv_transf_op, ::Nothing;
                kwargs...)
    # Pre-allocate for in-place back-transformation from eigenbasis
    rho_out = copy(transf_op)
    tmp = copy(transf_op)
    tmp2 = copy(transf_op)

    # Define fout
    function fout(t, rho)
        tmp.data[:] = rho.data
        QuantumOpticsBase.mul!(tmp2,transf_op,tmp)
        QuantumOpticsBase.mul!(rho_out,tmp2,inv_transf_op)
        return copy(rho_out)
    end

    return integrate(tspan, dmaster_br, copy(rho.data), rho, copy(rho), fout; kwargs...)
end

# Integrate with given fout
function integrate_br(tspan, dmaster_br, rho,
                transf_op, inv_transf_op, fout::F;
                kwargs...) where {F}
    # Pre-allocate for in-place back-transformation from eigenbasis
    rho_out = copy(transf_op)
    tmp = copy(transf_op)
    tmp2 = copy(transf_op)

    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)

    # Perform back-transfomration before calling fout
    function fout_(t, rho)
        tmp.data[:] = rho.data
        QuantumOpticsBase.mul!(tmp2,transf_op,tmp)
        QuantumOpticsBase.mul!(rho_out,tmp2,inv_transf_op)
        return fout(t, rho_out)
    end

    return integrate(tspan_, dmaster_br, copy(rho.data), rho, copy(rho), fout_; kwargs...)
end
