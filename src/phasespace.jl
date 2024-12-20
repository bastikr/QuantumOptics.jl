import WignerSymbols: clebschgordan
import QuantumOpticsBase:
    qfunc,
    wigner,
    coherentspinstate,
    qfuncsu2,
    wignersu2,
    BLROperator

"""
    qfunc(a, α)
    qfunc(a, x, y)
    qfunc(a, xvec, yvec)

Husimi Q representation ``⟨α|ρ|α⟩/π`` for the given state or operator `a`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`. Note that conversion from `x` and `y` to `α` is
done via the relation ``α = \\frac{1}{\\sqrt{2}}(x + i y)``.
"""
function qfunc(rho::BLROperator{B,B}, alpha) where B<:FockBasis
    b = basis(rho)
    _qfunc_operator(rho, alpha, Ket(b), Ket(b))
end

function qfunc(rho::BLROperator{B,B}, xvec::AbstractVector, yvec::AbstractVector) where B<:FockBasis
    b = basis(rho)
    Nx = length(xvec)
    Ny = length(yvec)
    tmp1 = Ket(b)
    tmp2 = Ket(b)
    result = Matrix{eltype(rho)}(undef, Nx, Ny)
    @inbounds for j=1:Ny, i=1:Nx
        result[i, j] = _qfunc_operator(rho, complex(xvec[i], yvec[j])/sqrt(2), tmp1, tmp2)
    end
    result
end

function qfunc(psi::Ket{B}, alpha) where B<:FockBasis
    b = basis(psi)
    _conj_alpha = conj(alpha)
    c = one(alpha)
    @inbounds for n=1:b.offset
        c *= _conj_alpha/sqrt(n)
    end
    s = c*psi.data[1]
    @inbounds for n=b.offset+1:b.N
        c *= _conj_alpha/sqrt(n)
        s += c*psi.data[n+1-b.offset]
    end
    return abs2(s)*exp(-abs2(_conj_alpha))/pi
end

function qfunc(psi::Ket{B}, xvec::AbstractVector, yvec::AbstractVector) where B<:FockBasis
    b = basis(psi)
    points = length(xvec)*length(yvec)
    N = length(b)::Int
    N0 = b.offset
    _conj_alpha = [complex(x, -y)/sqrt(2) for x=xvec, y=yvec]

    # Compute overlap <α|ψ> as reversed sum
    q = psi.data[N]/sqrt(b.N) .* _conj_alpha
    @inbounds for n=b.N:-1:N0+2
        x = psi.data[n-N0]
        f0_ = 1/sqrt(n-1)
        for i=1:points
            q[i] = (x + q[i])*_conj_alpha[i]*f0_
        end
    end

    # 1/sqrt(n!) up to offset for first term in sum
    nfac = 1.0
    @inbounds for n=1:N0
        nfac /= sqrt(n)
    end
    x = psi.data[1]
    @inbounds for i=1:points
        q[i] = (x + q[i])*nfac*_conj_alpha[i]^N0
    end

    # Return e^(-|α|^2)*|q|^2
    result = similar(q, float(real(eltype(psi))))
    @inbounds for i=1:points
        result[i] = abs2(q[i])*exp(-abs2(_conj_alpha[i]))/pi
    end
    return result
end

function qfunc(state::Union{Ket{B}, BLROperator{B,B}}, x, y) where B<:FockBasis
    qfunc(state, complex(x, y)/sqrt(2))
end

function _qfunc_operator(rho, alpha, tmp1, tmp2)
    coherentstate!(tmp1, basis(rho), alpha)
    QuantumOpticsBase.mul!(tmp2,rho,tmp1,true,false)
    a = dot(tmp1.data, tmp2.data)
    return a/pi
end


"""
    wigner(a, α)
    wigner(a, x, y)
    wigner(a, xvec, yvec)

Wigner function for the given state or operator `a`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`. Note that conversion from `x` and `y` to `α` is
done via the relation ``α = \\frac{1}{\\sqrt{2}}(x + i y)``.
"""
function wigner(rho::Operator{B,B}, x, y) where B<:FockBasis
    b = basis(rho)
    N = b.N
    N0 = b.offset
    _2α = complex(x, y)*sqrt(2)
    abs2_2α = abs2(_2α)
    w = complex(0.)
    coefficient = complex(0.)
    @inbounds for L=N:-1:1
        coefficient = 2*_clenshaw(L, abs2_2α, rho.data, N, N0)
        w = coefficient + w*_2α/sqrt(L+1)
    end
    coefficient = _clenshaw(0, abs2_2α, rho.data, N, N0)
    w = coefficient + w*_2α
    exp(-abs2_2α/2)/pi*real(w)
end

function wigner(rho::Operator{B,B}, xvec::AbstractVector, yvec::AbstractVector) where B<:FockBasis
    b = basis(rho)
    N = b.N
    N0 = b.offset
    _2α = [complex(x, y)*sqrt(2) for x=xvec, y=yvec]
    abs2_2α = abs2.(_2α)
    w = zero(_2α)
    b0 = similar(_2α)
    b1 = similar(_2α)
    b2 = similar(_2α)
    @inbounds for L=N:-1:1
        _clenshaw_grid!(w, L, rho.data, abs2_2α, _2α, b0, b1, b2, 2, N, N0)
    end
    _clenshaw_grid!(w, 0, rho.data, abs2_2α, _2α, b0, b1, b2, 1, N, N0)
    @inbounds for i=eachindex(w)
        abs2_2α[i] = exp(-abs2_2α[i]/2)/pi.*real(w[i])
    end
    abs2_2α
end

wigner(psi::Ket, x, y) = wigner(dm(psi), x, y)
wigner(state, alpha::Number) = wigner(state, real(alpha)*sqrt(2), imag(alpha)*sqrt(2))


function _clenshaw_grid!(w, L::Integer, ρ, abs2_2α, _2α, b0, b1, b2,
                            scale::Integer, N::Integer, offset::Integer)
    n = N-L
    points = length(w)
    if n==0
        if iszero(offset)
            f = scale*ρ[1, L+1]
        else
            f = zero(eltype(ρ))
        end
        @inbounds for i=1:points
            w[i] = f + w[i]*_2α[i]/sqrt(L+1)
        end
    elseif n==1
        f1 = 1/sqrt(L+1)
        if iszero(offset)
            @inbounds for i=1:points
                w[i] = scale*(ρ[1, L+1] - ρ[2, L+2]*(L+1-abs2_2α[i])*f1) + w[i]*_2α[i]*f1
            end
        elseif isone(offset)
            @inbounds for i=1:points
                w[i] = -scale*ρ[1, L+1]*(L+1-abs2_2α[i])*f1 + w[i]*_2α[i]*f1
            end
        else # offset > 1
            @inbounds for i=1:points
                w[i] *= _2α[i]*f1
            end
        end
    else
        f0 = sqrt(float((n+L-1)*(n-1)))
        f1 = sqrt(float((n+L)*n))
        f0_ = 1/f0
        f1_ = 1/f1
        if n < offset
            fill!(b1, zero(eltype(ρ)))
            fill!(b0, zero(eltype(ρ)))
        else
            fill!(b1, ρ[n+1-offset, L+n+1-offset])
            if n <= offset
                @inbounds for i=1:points
                    b0[i] = -(2*n-1+L-abs2_2α[i])*f1_*b1[i]
                end
            else
                @inbounds for i=1:points
                    b0[i] = ρ[n-offset, L+n-offset] - (2*n-1+L-abs2_2α[i])*f1_*b1[i]
                end
            end
        end

        @inbounds for k=n-2:-1:1
            b1, b2, b0 = b0, b1, b2
            if k < offset
                x = zero(eltype(ρ))
            else
                x = ρ[k+1-offset, L+k+1-offset]
            end
            a1 = -(2*k+1+L)
            a2 = -f0*f1_
            @inbounds for i=1:points
                b0[i] = x + (a1+abs2_2α[i])*f0_*b1[i] + a2*b2[i]
            end
            f1 , f1_ = f0, f0_
            f0 = sqrt((k+L)*k)
            f0_ = 1/f0
        end
        if iszero(offset)
            @inbounds for i=1:points
                w[i] = scale*(ρ[1, L+1] - (L+1-abs2_2α[i])*f0_*b0[i] - f0*f1_*b1[i]) + w[i]*_2α[i]*f0_
            end
        else
            @inbounds for i=1:points
                w[i] = scale*(-(L+1-abs2_2α[i])*f0_*b0[i] - f0*f1_*b1[i]) + w[i]*_2α[i]*f0_
            end
        end
    end
    return w
end

function _clenshaw(L::Integer, abs2_2α::Real, ρ, N::Integer, offset::Integer)
    n = N-L
    if n==0
        if iszero(offset)
            return ρ[1, L+1]
        else
            return zero(eltype(ρ))
        end
    elseif n==1
        if iszero(offset)
            ϕ1 = -(L+1-abs2_2α)/sqrt(L+1)
            return ρ[1, L+1] + ρ[2, L+2]*ϕ1
        elseif isone(offset)
            ϕ1 = -(L+1-abs2_2α)/sqrt(L+1)
            return ρ[1,L+1]*ϕ1
        else
            return zero(eltype(ρ))
        end
    else
        f0 = sqrt(float((n+L-1)*(n-1)))
        f1 = sqrt(float((n+L)*n))
        f0_ = 1/f0
        f1_ = 1/f1
        b2 = complex(0.)
        if n < offset
            b1 = zero(eltype(ρ))
        else
            b1 = ρ[n+1-offset, L+n+1-offset]
        end
        if n <= offset
            b0 = - (2*n-1+L-abs2_2α)*f1_*b1
        else
            b0 = ρ[n-offset, L+n-offset] - (2*n-1+L-abs2_2α)*f1_*b1
        end
        @inbounds for k=n-2:-1:1
            b1, b2 = b0, b1
            if k < offset
                b0 = - (2*k+1+L-abs2_2α)*f0_*b1 - f0*f1_*b2
            else
                b0 = ρ[k+1-offset, L+k+1-offset] - (2*k+1+L-abs2_2α)*f0_*b1 - f0*f1_*b2
            end
            f1, f1_ = f0, f0_
            f0 = sqrt((k+L)*k)
            f0_ = 1/f0
        end
        if iszero(offset)
            return ρ[1, L+1] - (L+1-abs2_2α)*f0_*b0 - f0*f1_*b1
        else
            return - (L+1-abs2_2α)*f0_*b0 - f0*f1_*b1
        end
    end
end

"""
    coherentspinstate(b::SpinBasis, θ::Real, ϕ::Real)

A coherent spin state |θ,ϕ⟩ is analogous to the coherent state of the linear harmonic
oscillator. Coherent spin states represent a collection of identical two-level
systems and can be described by two angles θ and ϕ (although this
parametrization is not unique), similarly to a qubit on the
Bloch sphere.
"""
function coherentspinstate(b::SpinBasis, theta::Real, phi::Real)

    result = Ket(b)
    data = result.data

    N = length(b)-1
    α = sin(theta / 2) * exp(1im * phi / 2)
    β = cos(theta / 2) * exp(-1im * phi / 2)

    # forward pass: `c_n = sqrt(binomial(N, n)) * α^n` with `n  ≥ 0`,
    # using recursive `binomial(N, n) = ((N+1-n)/n) * binomial(N, n-1)`
    coefficient = 1.0
    @inbounds for n = 1:N+1
        data[n] = coefficient
        coefficient *= α * sqrt((N + 1 - n) / n)
    end

    # backward pass:  c_n *= β^(N-n)
    factor = 1.0
    @inbounds for n = N+1:-1:1
        data[n] *= factor
        factor *= β
    end

    return result

end

"""
    qfuncsu2(ket,Ntheta;Nphi=2Ntheta)
    qfuncsu2(rho,Ntheta;Nphi=2Ntheta)

Husimi Q SU(2) representation ``⟨θ,ϕ|ρ|θ,ϕ⟩/π`` for the given state.

The function calculates the SU(2) Husimi representation of a state on the
generalised bloch sphere (0 < θ < π and 0 < ϕ < 2 π) with a given
resolution `(Ntheta, Nphi)`.

    qfuncsu2(rho,θ,ϕ)
    qfuncsu2(ket,θ,ϕ)

This version calculates the Husimi Q SU(2) function at a position given by θ and ϕ.
"""
function qfuncsu2(psi::Ket{B}, Ntheta::Integer; Nphi::Integer=2Ntheta) where B<:SpinBasis
    b = psi.basis
    psi_bra_data = psi.data'
    lb = float(b.spinnumber)
    result = Array{real(eltype(psi))}(undef, Ntheta,Nphi)
    @inbounds  for i = 0:Ntheta-1, j = 0:Nphi-1
        result[i+1,j+1] = (2*lb+1)/(4pi)*abs2(psi_bra_data*coherentspinstate(b,pi-i*pi/(Ntheta-1),j*2pi/(Nphi-1)-pi).data)
    end
    return result
end

function qfuncsu2(rho::Operator{B,B}, Ntheta::Integer; Nphi::Integer=2Ntheta) where B<:SpinBasis
    b = basis(rho)
    lb = float(b.spinnumber)
    result = Array{real(eltype(rho))}(undef, Ntheta,Nphi)
    @inbounds  for i = 0:Ntheta-1, j = 0:Nphi-1
        c = coherentspinstate(b,pi-i*1pi/(Ntheta-1),j*2pi/(Nphi-1)-pi)
        result[i+1,j+1] = abs((2*lb+1)/(4pi)*c.data'*rho.data*c.data)
    end
    return result
end

function qfuncsu2(psi::Ket{B}, theta::Real, phi::Real) where B<:SpinBasis
    b = basis(psi)
    psi_bra_data = psi.data'
    lb = float(b.spinnumber)
    result = (2*lb+1)/(4pi)*abs2(psi_bra_data*coherentspinstate(b,theta,phi).data)
    return result
end

function qfuncsu2(rho::Operator{B,B}, theta::Real, phi::Real) where B<:SpinBasis
    b = basis(rho)
    lb = float(b.spinnumber)
    c = coherentspinstate(b,theta,phi)
    result = abs((2*lb+1)/(4pi)*c.data'*rho.data*c.data)
    return result
end

"""
    wignersu2(ket,Ntheta;Nphi=2Ntheta)
    wignersu2(rho,Ntheta;Nphi=2Ntheta)

Wigner SU(2) representation for the given state with a resolution `(Ntheta, Nphi)`.

The function calculates the SU(2) Wigner representation of a state on the
generalised bloch sphere (0 < θ < π and 0 < ϕ < 2 π) with a given resolution by
decomposing the state into the basis of spherical harmonics.

    wignersu2(rho,θ,ϕ)
    wignersu2(ket,θ,ϕ)

This version calculates the Wigner SU(2) function at a position given by θ and ϕ
"""
function wignersu2(rho::Operator{B,B}, theta::Real, phi::Real) where B<:SpinBasis

    N = length(basis(rho))-1

    ### Tensor generation ###
    BandT = Array{Vector{real(eltype(rho))}}(undef, N,N+1)
    BandT[1,1] = collect(range(-N/2, stop=N/2, length=N+1))
    BandT[1,2] = -collect(sqrt.(range(1, stop=N, length=N)).*sqrt.(range((N)/2, stop=1/2, length=N)))
    BandT[2,1] = clebschgordan(1,0,1,0,2,0)*BandT[1,1].*BandT[1,1] -
        clebschgordan(1,-1,1,1,2,0)*[zeros(N+1-length(BandT[1,2])); BandT[1,2].*BandT[1,2]] -
        clebschgordan(1,1,1,-1,2,0)*[BandT[1,2].*BandT[1,2]; zeros(N+1-length(BandT[1,2]))]
    BandT[2,2] = clebschgordan(1,0,1,1,2,1)BandT[1,1][1:N].*BandT[1,2]+
        clebschgordan(1,1,1,0,2,1)*BandT[1,2][1:N].*BandT[1,1][2:end]
    BandT[2,3] = BandT[1,2][1:N+1-(2)].*BandT[1,2][2:end]

    @inbounds for S=2:N-1
        BandT[S+1,1] = clebschgordan(1,0,S,0,S+1,0)*BandT[1,1].*BandT[S,1] -
            [zeros(N+1-length(BandT[1,2])); clebschgordan(1,-1,S,1,S+1,0)*BandT[1,2].*BandT[S,2]] -
            clebschgordan(1,1,S,-1,S+1,0)*[BandT[1,2].*BandT[S,2]; zeros(N+1-length(BandT[1,2]))]
        BandT[S+1,S+1] = clebschgordan(1,0,S,S,S+1,S)BandT[1,1][1:N+1-S].*BandT[S,S+1]+
            clebschgordan(1,1,S,S-1,S+1,S)*BandT[1,2][1:N+1-S].*BandT[S,S][2:end]
        BandT[S+1,S+2] = BandT[1,2][1:N+1-(S+1)].*BandT[S,S+1][2:end]
        @inbounds for M=1:S-1
            BandT[S+1,M+1] = clebschgordan(1, 0, S, M, S+1,M)*BandT[1,1][1:N+1-M].*BandT[S,M+1] +
                clebschgordan(1,1,S,M-1,S+1,M)*BandT[1,2][1:N+1-M].*BandT[S,M][2:end] -
                clebschgordan(1,-1,S,M+1,S+1,M)*[zeros(1); BandT[1,2][1:N-M].*BandT[S,M+2][1:N-M]]
        end

    end

    NormT = zeros(N)
    @inbounds for S = 1:N
        NormT[S] = sum(BandT[S,1].^2)
    end

    @inbounds for S = 1:N, M = 0:S
        BandT[S, M + 1] =  BandT[S, M + 1]/sqrt(NormT[S])
    end

    ### State decomposition ###
    c = rho.data
    EVT = Array{ComplexF64}(undef, N,N+1)
    @inbounds for S = 1:N, M = 0:S
        EVT[S,M+1] = conj(sum(BandT[S,M+1].*diag(c,M)))
    end

    wignermap = _wignersu2int(N,theta,phi, EVT)
    return wignermap*sqrt((N+1)/(4pi))
end

function wignersu2(rho::Operator{B,B}, Ntheta::Integer; Nphi::Integer=2Ntheta) where B<:SpinBasis

    N = length(basis(rho))-1

    ### Tensor generation ###
    BandT = Array{Vector{real(eltype(rho))}}(undef, N,N+1)
    BandT[1,1] = collect(range(-N/2, stop=N/2, length=N+1))
    BandT[1,2] = -collect(sqrt.(range(1, stop=N, length=N)).*sqrt.(range((N)/2, stop=1/2, length=N)))
    BandT[2,1] = clebschgordan(1,0,1,0,2,0)*BandT[1,1].*BandT[1,1] -
        clebschgordan(1,-1,1,1,2,0)*[zeros(N+1-length(BandT[1,2])); BandT[1,2].*BandT[1,2]] -
        clebschgordan(1,1,1,-1,2,0)*[BandT[1,2].*BandT[1,2]; zeros(N+1-length(BandT[1,2]))]
    BandT[2,2] = clebschgordan(1,0,1,1,2,1)BandT[1,1][1:N].*BandT[1,2]+
        clebschgordan(1,1,1,0,2,1)*BandT[1,2][1:N].*BandT[1,1][2:end]
    BandT[2,3] = BandT[1,2][1:N+1-(2)].*BandT[1,2][2:end]

    @inbounds for S=2:N-1
        BandT[S+1,1] = clebschgordan(1,0,S,0,S+1,0)*BandT[1,1].*BandT[S,1] -
            [zeros(N+1-length(BandT[1,2])); clebschgordan(1,-1,S,1,S+1,0)*BandT[1,2].*BandT[S,2]] -
            clebschgordan(1,1,S,-1,S+1,0)*[BandT[1,2].*BandT[S,2]; zeros(N+1-length(BandT[1,2]))]
        BandT[S+1,S+1] = clebschgordan(1,0,S,S,S+1,S)BandT[1,1][1:N+1-S].*BandT[S,S+1]+
            clebschgordan(1,1,S,S-1,S+1,S)*BandT[1,2][1:N+1-S].*BandT[S,S][2:end]
        BandT[S+1,S+2] = BandT[1,2][1:N+1-(S+1)].*BandT[S,S+1][2:end]
        @inbounds for M=1:S-1
            BandT[S+1,M+1] = clebschgordan(1, 0, S, M, S+1,M)*BandT[1,1][1:N+1-M].*BandT[S,M+1] +
                clebschgordan(1,1,S,M-1,S+1,M)*BandT[1,2][1:N+1-M].*BandT[S,M][2:end] -
                clebschgordan(1,-1,S,M+1,S+1,M)*[0.0im; BandT[1,2][1:N-M].*BandT[S,M+2][1:N-M]]
        end
    end

    NormT = zeros(N)
    @inbounds for S = 1:N
        NormT[S] = sum(BandT[S,1].^2)
    end
    @inbounds for S = 1:N, M = 0:S
        BandT[S, M + 1] = BandT[S, M + 1]/sqrt(NormT[S])
    end

    ### State decomposition ###
    c = rho.data
    EVT = Array{ComplexF64}(undef, N,N+1)
    @inbounds for S = 1:N, M = 0:S
        EVT[S,M+1] = conj(sum(BandT[S,M+1].*diag(c,M)))
    end

    wignermap = Array{Float64}(undef, Ntheta,Nphi)
    @inbounds for i = 1:Ntheta, j = 1:Nphi
        wignermap[i,j] = _wignersu2int(N,i*1pi/(Ntheta-1)-1pi/(Ntheta-1),j*2pi/(Nphi-1)-2pi/(Nphi-1)-pi, EVT)
    end
    return wignermap*sqrt((N+1)/(4pi))
end

function _wignersu2int(N::Integer, theta::Real, phi::Real, EVT::Array{ComplexF64, 2})
    UberBand = sqrt(1/(1+N))*ylm(0,0,theta,phi)
    @inbounds for S = 1:N
        @inbounds for M = 1:S
            UberBand += 2*real(EVT[S,M+1]*conj(ylm(S,M,theta,phi)))
        end
        UberBand += EVT[S,1]*ylm(S,0,theta,phi)
    end
    UberBand
end
wignersu2(psi::Ket, args...) = wignersu2(dm(psi), args...)

"""
    ylm(l::Integer,m::Integer,theta::Real,phi::Real)

Spherical harmonics Y(l,m)(θ,ϕ) where l ∈ N,  m = -l,-l+1,...,l-1,l, θ ∈ [0,π],
and ϕ ∈ [0,2π).

This function calculates the value of Y(l,m) spherical harmonic at position θ and ϕ.
"""
function ylm(l::Integer,m::Integer,theta::Real,phi::Real)
    phi_ = mod(phi,2pi)
    theta_ = mod(theta,2pi)
    phase = exp(1.0im*m*phi_)
    if theta_ ≈ 0
        if m == 0
            return @. phase*sqrt((2*l+1)/pi)/2
        else
            return 0
        end
    elseif theta_ ≈ pi
        if m == 0
            return @. phase*(-1)^l*sqrt((2*l+1)/pi)/2
        else
            return 0
        end
    else
        if l == 0
            return 1.0/sqrt(4pi)
        else
            m_ = abs(m)
            norm = _calc_ylm_norm(l, m_)
            sign = m > 0 ? (-1)^m_ : 1
            arg = cos(theta_)
            p_ll = 1.0
            @inbounds for fact = 1.0:l
                p_ll *= @. 1.0/((2*fact))*sqrt(1-arg^2)
            end

            if m_ == l
                return @. p_ll/norm*phase*sign
            elseif l-m_ == 1
                p_llp1 = @. 2*l*arg/sqrt(1-arg^2)*p_ll
                return @. p_llp1/norm*phase*sign
            else
                p_llp1 = @. 2*l*arg/sqrt(1-arg^2)*p_ll
                @inbounds for mr = -l:-m_-2
                    p_llp2 = @. -2*(mr+1)*arg/sqrt(1-arg^2)*p_llp1-(l-mr)*(l+mr+1)*p_ll
                    p_ll = p_llp1
                    p_llp1 = p_llp2
                end
                return @. p_llp1/norm*phase*sign
            end
        end
    end
end

function _calc_ylm_norm(l, m)
    n = sqrt(4pi/(2l+1))
    for k=(-m+1):m
        n /= sqrt(l + k)
    end
    return n
end
