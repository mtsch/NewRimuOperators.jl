# To-do:
# * The length of the DFT vectors is generally not the same as the number of modes, so 
#   I need to be careful about which is used as a type parameter
# * two-body term should really be an interaction matrix, which does not play well with MomPotentialFunction;
#   this may go back to how to deal with mixed term: u'(x-y) v'(x), where x and y may be of different components
# * optimise SFunction


struct SFunction{P}
    values::SVector{P,Float64}
end
function SFunction(M; pad=2)
    P = 2M - isodd(M)           # range of k
    Q = pad*2M - isodd(M)       # range of sum over k'
    dft = momentum_space_harmonic_potential(Q, 2pad)
    ns = range(-Q + isodd(M); length=Q) # [-pM, pM) including left boundary
    ks = n_to_k.(shift_lattice(ns), Q)
    kvk = ks .* dft

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
    dft = momentum_space_harmonic_potential(P, 2)   # factor of 2 is due to 1/M in definition of `momentum_space_harmonic_potential`
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
    dft = momentum_space_harmonic_potential(P, 2)   # factor of 2 is due to 1/M in definition of `momentum_space_harmonic_potential`
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
    v::V
    b::V
    pad::Int
    twobody::Bool
end
function TranscorrelatedPotential(v; b=1, pad=2, twobody=false)
    return TranscorrelatedPotential{typeof(v)}(v, b, pad, twobody)
end

function initialize(tcp::TranscorrelatedPotential, ham)
    if basis(ham) ≡ MomentumSpace()
        address = starting_address(ham)
        M = num_modes(address)
        u = only(ham.u) / only(ham.t)
        pad = tcp.pad

        v = tcp.v   # trap strengths
        b = tcp.b   # Jastrow widths

        B = @. -1/4b^2                              # reconcile trap, interaction and Jastrow parameters
        c1 = parameter_column(address, B)           # coefficient of the quadratic and non-hermitian terms
        c2 = parameter_column(address, @. 4B^2 * v) # coefficient of the derivative-squared term
        
        term = FullOneBodyTerm(TCPotentialOneBody(M, c1, c2, pad))
        # term += HarmonicPotential(v)
        
        if tcp.twobody && u ≠ 0
            cutoff = ham isa Transcorrelated ? ham.cutoff : @warn "Using `TranscorrelatedPotential` with a bare Hamiltonian"
            c3 = parameter_column(address, 2u .* B)      # this actually needs to be an interaction_matrix
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
