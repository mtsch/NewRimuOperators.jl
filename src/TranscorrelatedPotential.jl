struct SFunction{M}
    values::SVector{M,Float64}
end
function SFunction(M; pad=2)
    N = 2M - isodd(M)           # range of k
    bigN = pad*2M - isodd(M)    # range of sum over k'
    dft = momentum_space_harmonic_potential(bigN, pad)
    ns = range(-bigN + isodd(M); length=bigN) # [-M, M) including left boundary
    ks = n_to_k.(shift_lattice(ns), bigN)
    kvk = ks .* dft

    s = SVector{N}([dot(kvk, circshift(kvk, -j)) for j in 0:N-1])   # need to optimise this
    return SFunction{M}(s)
end
(s::SFunction)(n::Int) = s.values[abs(n) + 1]

"""
TCPotentialOneBody(M, cutoff, v, t)

The function used in the one-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialOneBody{M,V}
    corr_v::MomPotentialFunction{M,Float64}
    s::SFunction{M}
    v::V
    b::Float64
end
function TCPotentialOneBody(M, v, b, pad)
    dft = momentum_space_harmonic_potential(2M, 2)
    corr_v = MomPotentialFunction(1, dft)
    s = SFunction(M; pad)
    return TCPotentialOneBody(corr_v, s, v, b)
end
function (f::TCPotentialOneBody{M})(σ, p, q) where {M}
    k = q - p
    k_pi = n_to_k(k, M)
    p_pi = i_to_k(p, M) # NB: input momenta are in 1:M mode index form
    vk = f.corr_v(k)
    s_k = f.s(k)
    B = -1/4f.b^2
    v = f.v(σ)

    return B * (k_pi^2 + 2p_pi * k_pi) * vk + 4B^2 * v * s_k
end

"""
    TCPotentialTwoBody(M, cutoff, v, t)

The function used in the two-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialTwoBody{M,V}
    num_modes::Int
    corr_u::CorrelationFactor{M}
    corr_v::MomPotentialFunction{M,Float64}
    u::Float64
end
function TCPotentialTwoBody(M, cutoff::Int, u, pad)
    corr_u = CorrelationFactor(M, cutoff; length=2M)
    corr_v = MomPotentialFunction(1, momentum_space_harmonic_potential(M, 1))
    TCPotentialTwoBody(M, corr_u, corr_v, u)
end
function (f::TCPotentialTwoBody)(σ, τ, s, r, q, p)
    return f.u * f.corr_v(p + q - r - s) * n_to_k(p + q - r - s) * f.corr_u(r - q) * n_to_k(r - q)
end

"""
    TranscorrelatedPotential(v; cutoff, twobody=false)

Transcorrelated external potential with strength `v`. The argument `v` can be a real
number or a vector of real numbers whose length matches the number of components in the
Hamiltonian.

The `cutoff` defaults to the cutoff of the Hamiltonian is attached to, or 1.

Can only be used with momentum space Hamiltonians.
"""
struct TranscorrelatedPotential{V} <: ExtensionPrototype
    v::V
    b::Float64
    pad::Int
    twobody::Bool
end
function TranscorrelatedPotential(v; b=1, pad=2, twobody=false)
    return TranscorrelatedPotential(v, b, pad, twobody)
end

function initialize(tcp::TranscorrelatedPotential, ham)
    if basis(ham) ≡ MomentumSpace()
        address = starting_address(ham)
        M = num_modes(address)
        v = parameter_column(address, tcp.v)
        b = tcp.b
        u = only(ham.u) / only(ham.t)
        pad = tcp.pad
        
        term = FullOneBodyTerm(TCPotentialOneBody(M, v, b, pad))
        term += HarmonicPotential(v)
        
        if tcp.twobody && u ≠ 0
            cutoff = ham isa Transcorrelated ? ham.cutoff : 1
            return term + FullTwoBodyTerm(TCPotentialTwoBody(M, cutoff, -u/4b^2))
        else
            return term
        end
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `TranscorrelatedPotential`"
        ))
    end
end
