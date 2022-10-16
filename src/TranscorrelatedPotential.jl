# To-do:
# * The length of the DFT vectors is generally not the same as the number of modes, so 
#   I need to be careful about which is used as a type parameter
# * two-body term should really be an interaction matrix, which does not play well with MomPotentialFunction;
#   this may go back to how to deal with mixed term: u'(x-y) v'(x), where x and y may be of different components
# * optimise SFunction

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
function trap_dft(M::Integer)
    # Set up potential like in Real1DEP
    is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
    js = shift_lattice(is) # shifted such that js[1] = 0
    real_potential = [j^2 for j in js]
    mom_potential = fft(real_potential)
    # This should never fail for a harmonic oscillator, but it's best to check just in case.
    for x in mom_potential
        @assert iszero(real(x)) || abs(imag(x) / real(x)) < sqrt(eps())
    end
    # Make sure it's completely symmetric. It should be, but round-off errors can sometimes
    # make it non-symmetric.
    for i in 1:M÷2
        mom_potential[M - i + 1] = mom_potential[i + 1]
    end
    return SVector{M}(real.(mom_potential))
end


struct SFunction{P}
    values::SVector{P,Float64}
end
function SFunction(M; pad=2)
    P = 2M - isodd(M)           # range of k
    Q = pad*2M - isodd(M)       # range of sum over k'
    dft = trap_dft(Q)
    ns = range(-Q + isodd(M); length=Q) # [-pM, pM) including left boundary
    ks = n_to_k.(shift_lattice(ns), Q)
    kvk = ks .* dft ./ Q

    s = SVector{P}([dot(kvk, circshift(kvk, -j)) for j in 0:P-1])   # need to optimise this
    return SFunction{P}(s)
end
(s::SFunction)(n::Int) = s.values[abs(n) + 1]

"""
TCPotentialOneBody(M, c1, c2, pad)

The function used in the one-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialOneBody{M,P,V}
    corr_v::MomPotentialFunction{P,V}
    s::SFunction{P}
    coeff2::V
end
function TCPotentialOneBody(M, c1, c2, pad)
    P = 2M - isodd(M)           # range of k
    dft = trap_dft(P)
    corr_v = MomPotentialFunction(c1, dft)
    s = SFunction(M; pad)
    return TCPotentialOneBody{M,P,typeof(c2)}(corr_v, s, c2)
end
function (f::TCPotentialOneBody{M})(σ, p, q) where {M}
    k = q - p
    k_pi = n_to_k(k, M)
    p_pi = i_to_k(p, M) # NB: input momenta are in 1:M mode index form
    vk = f.corr_v(σ, p, q)
    s_k = f.s(k)
    # B = -1/4f.b^2
    # v = f.v(σ)
    # A = f.A(σ)
    B = f.coeff2(σ)

    return (k_pi^2 + 2p_pi * k_pi) * vk + B * s_k
end

"""
    TCPotentialTwoBody(M, cutoff, c)

The function used in the two-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialTwoBody{P,V}
    num_modes::Int
    corr_u::CorrelationFactor{P}
    corr_v::MomPotentialFunction{P,ConstFunction{Int}}
    coeff::V
end
function TCPotentialTwoBody(M, cutoff::Int, c)
    P = 2M - isodd(M)           # range of k and k'
    dft = 2 .* trap_dft(P)  # factor of 2 is due to 1/M in definition of `momentum_space_harmonic_potential`
    corr_v = MomPotentialFunction(ConstFunction{Int}(1), dft)   # this seems janky;
    corr_u = CorrelationFactor(M, cutoff; length=P)
    TCPotentialTwoBody(M, corr_u, corr_v, c)
end
function (f::TCPotentialTwoBody)(σ, τ, s, r, q, p)
    M = f.num_modes
    kp = r - q
    k = s - p + kp
    u = f.coeff(σ, τ)
    return u * f.corr_v(1, k, 0) * n_to_k(kp, M) * f.corr_u(kp) * n_to_k(kp, M)
end

"""
    TranscorrelatedPotential(v; b=1, pad=2, twobody=false)

Transcorrelated external potential with strength `v`. 

* `v` can be a real number or a vector of real numbers whose length matches the number of components in the
Hamiltonian.
* `b` is the width of the Jastrow factor relative to the trap length.
* `pad` scales the number of terms to use in internal sums relative to the number of modes
* `twobody` toggles the two body terms

Can only be used with momentum space Hamiltonians.
"""
struct TranscorrelatedPotential{V} <: ExtensionPrototype
    # v::V
    b::V
    pad::Int
    twobody::Bool
end
function TranscorrelatedPotential(; b=1, pad=2, twobody=false)
    return TranscorrelatedPotential{typeof(b)}(b, pad, twobody)
end

function initialize(tcp::TranscorrelatedPotential, ham)
    if basis(ham) ≡ MomentumSpace()
        address = starting_address(ham)
        M = num_modes(address)
        u = only(ham.u) 
        t = only(ham.t)
        pad = tcp.pad

        # v = tcp.v   # trap strengths
        b = tcp.b   # Jastrow widths

        B = @. -1/4b^2                              # reconcile trap, interaction and Jastrow parameters
        c1 = parameter_column(address, B)           # coefficient of the quadratic and non-hermitian terms
        c2 = parameter_column(address, @. B^2 / t)  # coefficient of the derivative-squared term
        
        term = FullOneBodyTerm(TCPotentialOneBody(M, c1, c2, pad))
        # term += HarmonicPotential(v)
        
        if tcp.twobody && u ≠ 0
            cutoff = ham isa Transcorrelated ? ham.cutoff : @warn "Using `TranscorrelatedPotential` with a bare Hamiltonian"
            c3 = parameter_column(address, @. 2B * u/t)      # this actually needs to be an interaction_matrix
            return term + FullTwoBodyTerm(TCPotentialTwoBody(M, cutoff, c3))
        else
            return term
        end
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `TranscorrelatedPotential`"
        ))
    end
end
