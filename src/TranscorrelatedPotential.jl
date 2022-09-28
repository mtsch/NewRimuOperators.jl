struct SFunction{M}
    s::SVector{M,Float64}
end
function SFunction(M)
    dft = MomPotentialFunction(1, momentum_space_harmonic_potential(M, 1))
    s = zeros(M,M)
    ks = n_to_k.(0:M-1,M)
    for i in 1:M, j in 1:M
        ks[i] * dft[i] * ks[j-i] * dft[j-i]
    end
    return SFunction{M}(SVector{M}(sum(s; dim=2)))
end
function (f::SFunction{M})(σ, p, q) where {M}
    return f.v(σ) * f.potential[mod(p - q, M) + 1]
end

"""
TCPotentialOneBody(M, cutoff, v, t)

The function used in the one-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialOneBody{M,V}
    corr::MomPotentialFunction{M,V}
    s::SFunction{M}
    v::V
end

function TCPotentialOneBody(M::Int, v)
    corr = MomPotentialFunction(1, momentum_space_harmonic_potential(M, 1))
    s = SFunction(M)
    return TCPotentialOneBody(corr, s, v)
end

function (f::TCPotentialOneBody{M})(σ, p, q) where {M}
    bsq = f.bsq
    v = d.v(σ)
    k = q - p
    k_pi = n_to_k(k, M)
    p_pi = i_to_k(p, M) # why is this different?
    corr_k = f.corr(k)

    result = -2/bsq - 4/bsq * (corr_k * k_pi + 2 * corr_k * p_pi)
    if bsq ≠ 1/2
        result += (1 - 4/bsq^2) * f.s(k)
    end
    return result
end

"""
    TCPotentialTwoBody(M, cutoff, v, t)

The function used in the two-body term of the transcorrelated harmonic trap potential.
"""
struct TCPotentialTwoBody{M,A}
    num_modes::Int
    corr_u::CorrelationFactor{M}
    corr_v::MomPotentialFunction{M,Float64}
    a::A
end
function TCPotentialTwoBody(M, cutoff::Int, t, u, v)
    ufunc = CorrelationFactor(M, cutoff; length=2M)
    vfunc = MomPotentialFunction(1, momentum_space_harmonic_potential(M, 1))
    TCPotentialTwoBody(M, ufunc, vfunc, u*v)
end

function (f::TCPotentialTwoBody)(σ, τ, s, r, q, p)
    return f.a[σ,τ] * f.corr_v(p + q - r - s) * (p + q - r - s) * f.corr_u(r - q) * (r - q)
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
    bsq::Float64
    twobody::Bool
end
function TranscorrelatedPotential(v; bsq=1/2, twobody=false)
    return TranscorrelatedPotential(v, bsq, twobody)
end

function initialize(tcp::TranscorrelatedPotential, ham)
    if basis(ham) ≡ MomentumSpace()
        address = starting_address(ham)
        M = num_modes(address)
        t = only(ham.t)
        u = only(ham.u)
        cutoff = ham isa Transcorrelated ? ham.cutoff : 1
        v = parameter_column(address, tcp.v)

        term = FullOneBodyTerm(TCPotentialOneBody(M, cutoff, v))
        if tcp.twobody && u ≠ 0
            return term + FullTwoBodyTerm(TCPotentialTwoBody(M, cutoff, u, v))
        else
            return term
        end
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `TranscorrelatedPotential`"
        ))
    end
end
