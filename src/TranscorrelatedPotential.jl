"""
    trap_dft(M::Integer, v_ho::Real) -> w

Set up a harmonic potential for use with momentum space Hamiltonians:
```math
    w(n) = \\mathrm{DFT}[j^2]_n,
```
where
```math
	j = \\left \\{
	\\begin{array}{lll}
		j' + \\frac{M}{2} + 1, & j' \\in [-\\frac{M}{2}, \\frac{M}{2}] & M \\text{ even} \\
		j' + \\frac{M-1}{2} + 1, & j' \\in [-\\frac{M-1}{2}, \\frac{M-1}{2}] & M \\text{ odd},
	\\end{array}
	\\right.
```
and ``\\mathrm{DFT}[…]_n`` is a discrete Fourier transform performed by `fft()[n%M + 1]`.
"""
function trap_dft(M::Int)
    # Set up potential like in Real1DEP
    is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
    js = shift_lattice(is) # shifted such that js[1] = 0
    real_potential = [j^2 for j in js]
    mom_potential = fft(real_potential)
    # This should never fail for a harmonic oscillator, but it's best to check just in case.
    for x in mom_potential
        @assert iszero(real(x)) || abs(imag(x) / real(x)) < sqrt(eps())
    end
    # enforce symmetry
    for i in 1:M÷2
        mom_potential[M - i + 1] = mom_potential[i + 1]
    end
    # SVector not necessary
    return SVector{M,Float64}(real.(mom_potential))
end

"""
    STensor(M, v)

Compute the tensor
```math
    S_k = \\sum_{k'} k' \\, (k-k') \\tilde{v}(k') \\tilde{v}(k-k'),
```
where ``\\tilde{v}(k)`` is the DFT of a harmonic trapping potential on a lattice of ``M`` sites.
The values of `k` are in ``\\frac{2\\pi}{M\\alpha}[-M+1, M-1]``.
"""
struct STensor{P}
    values::SVector{P,Float64}
end
function STensor(M, v::SVector{Q,Float64}) where Q
    P = 2M - 1
    z = (Q + 1) ÷ 2M    # Q == 2zM - 1
    
    n_range = -M+1:M-1              # k values
    np_range = -z*M+1:z*M-1         # k' values

    s = zeros(P)
    for n in n_range
        for np in np_range
            s[n + M] += (2π/M)^2 * np * (n - np) * v[1 + mod(np, Q)] * v[1 + mod(n - np, Q)]
        end
    end
    # s = [sum((2π/M)^2 * np * (n - np) * v[1 + mod(n, Q)] * v[1 + mod(n - np, Q)] for np in np_range) for n in n_range]
    return STensor{P}(SVector{P}(s))
end
(s::STensor{P})(n::Int) where {P} = s.values[n + (P + 1) ÷ 2]   # (P + 1) ÷ 2 == M for any M

"""
    TCPotentialOneBody(M, c1, c2, v)

The function used in the one-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialOneBody{M,P,V}
    corr_v::MomPotentialFunction{P,V}
    s::STensor{P}
    coeff2::V
end
function TCPotentialOneBody(M, c1, c2, v)
    P = 2*M - 1             # range of k
    # Q = 2*pad*M - 1         # range of k'
    # dft = trap_dft(Q) / (pad*M)^3   # magic factor ≈ (zM)^3
    corr_v = MomPotentialFunction(c1, SVector{P}(v[1:P]))
    s = STensor(M, v)
    return TCPotentialOneBody{M,P,typeof(c2)}(corr_v, s, c2)
end
function (f::TCPotentialOneBody{M})(σ, p, q) where {M}
    k = q - p
    k_pi = n_to_k(k, M)
    p_pi = i_to_k(p, M) # NB: input momenta are in 1:M mode index form
    v_k = f.corr_v(σ, p, q)
    s_k = f.s(k)
    c2 = f.coeff2(σ)

    return (k_pi^2 + 2p_pi * k_pi) * v_k + c2 * s_k
end

"""
    TCPotentialTwoBody(M, v, cutoff, c)

The function used in the two-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialTwoBody{P,V}
    num_modes::Int
    corr_u::CorrelationFactor{P}
    corr_v::MomPotentialFunction{P,ConstFunction{Int}}
    coeff::V
end
function TCPotentialTwoBody(M, v, cutoff::Int, c3)
    P = 2M - 1           # range of k and k'
    corr_v = MomPotentialFunction(ConstFunction{Int}(1), SVector{P}(v[1:P]))
    corr_u = CorrelationFactor(M, cutoff; length=P)
    TCPotentialTwoBody(M, corr_u, corr_v, c3)
end
function (f::TCPotentialTwoBody)(σ, τ, s, r, q, p)
    M = f.num_modes
    kp = r - q
    k = s - p + kp
    v_k = f.corr_v(σ, k, 0)
    u_k = f.corr_u(kp)
    c3 = f.coeff(σ, τ)
    return c3 * v_k * n_to_k(kp, M) * u_k * n_to_k(kp, M)
end

"""
    TranscorrelatedPotential(v; b=1, pad=1, twobody=false)

Transcorrelated external potential with strength `v`. 

* `v` can be a real number or a vector of real numbers whose length matches the number of components in the
Hamiltonian.
* `b` is the width of the Jastrow factor relative to the trap length.
* `pad` scales the number of terms to use in internal sums relative to the number of modes, namely 
    ``k' \\in \\frac{2\\pi}{M\\alpha}[-pad*M+1, pad*M-1]``.
* `twobody` toggles the two body terms involving the interparticle interaction.

Can only be used with momentum space Hamiltonians.
"""
struct TranscorrelatedPotential{V} <: ExtensionPrototype
    b::V
    pad::Int
    twobody::Bool
end
function TranscorrelatedPotential(; b=1, pad=1, twobody=false)
    return TranscorrelatedPotential{typeof(b)}(b, pad, twobody)
end

function initialize(tcp::TranscorrelatedPotential, ham)
    if basis(ham) ≡ MomentumSpace()
        address = starting_address(ham)
        M = num_modes(address)
        u = only(ham.u) 
        t = only(ham.t)
        pad = tcp.pad

        b = tcp.b   # Jastrow widths

        B = @. -1/4b^2                              # reconcile trap, interaction and Jastrow parameters
        c1 = parameter_column(address, B)           # coefficient of the quadratic and non-hermitian terms
        c2 = parameter_column(address, @. B^2 / t)  # coefficient of the derivative-squared term

        # DFT
        # P = 2*M - 1             # range of k
        Q = 2*pad*M - 1         # range of k'
        v_k = trap_dft(Q) / (pad*M)^3   # magic factor ≈ (zM)^3
        
        term = FullOneBodyTerm(TCPotentialOneBody(M, c1, c2, v_k))
        
        if tcp.twobody && u ≠ 0
            cutoff = ham isa Transcorrelated ? ham.cutoff : @warn "Using `TranscorrelatedPotential` with a bare Hamiltonian"
            c3 = interaction_matrix(address, @. 2B * u/t)
            return term + FullTwoBodyTerm(TCPotentialTwoBody(M, v_k, cutoff, c3))
        else
            return term
        end
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `TranscorrelatedPotential`"
        ))
    end
end
