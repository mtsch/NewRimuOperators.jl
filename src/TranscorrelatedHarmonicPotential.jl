"""
TCTrapOneBody(M, cutoff, v, t)

The function used in the one-body term of the transcorrelated harmonic trap potential.
"""
struct TCTrapOneBody{M,V}
    corr::CorrelationFactor{M}
    w::WFunction{M}
    v::V
    t::Float64
end

function TCTrapOneBody(M::Int, cutoff::Int, v, t)
    corr = CorrelationFactor(M, cutoff)
    w = WFunction(M, cutoff)
    return TCDeltaFunctionOneBody(corr, w, v, t)
end

function (d::TCTrapOneBody{M})(σ, p, q) where {M}
    t = d.t
    v = d.v(σ)
    k = q - p
    k_pi = n_to_k(k, M)
    p_pi = i_to_k(p, M)
    cor_k = d.corr(k)

    # note: the factors of 2 and 4 come from the fact that ṽ does not have a factor of
    # 2 in the denominator.
    return v/M + 2v/M * (cor_k * k_pi + 2 * cor_k * p_pi) + 4*v^2/t * d.w(k)
end

"""
    TCTrapTwoBody(M, cutoff, v, t)

The function used in the two-body term of the transcorrelated harmonic trap potential.
"""
struct TCTrapTwoBody{M,V}
    num_modes::Int
    corr::CorrelationFactor{M}
    v::V
    u::Float64
    t::Float64
end
function TCTrapTwoBody(M, cutoff::Int, v::V, u, t) where {V}
    TCTrapTwoBody(M, CorrelationFactor(M, cutoff; length=2M), v, u, t)
end

function (d::TCTrapTwoBody)(σ, τ, s, r, q, p)
    M = d.num_modes; u = d.u; v = d.v; t = d.t; corr = d.corr
    result_στ = (v(σ) * corr(p + q - r - s) * corr(r - q))
    result_τσ = (v(τ) * corr(p + q - r - s) * corr(s - p))
    return 4u/(t*M^2) * (result_στ + result_τσ)
end

"""
    TranscorrelatedHarmonicPotential(v; cutoff, two_body_term=false)

Transcorrelated harmonic trap potential with strength `v`. The argument `v` can be a real
number or a vector of real numbers whose length matches the number of components in the
Hamiltonian.

The `cutoff` defaults to the cutoff of the Hamiltonian is attached to, or 1.

Can only be used with momentum space Hamiltonians.
"""
struct TranscorrelatedHarmonicPotential{V,C} <: ExtensionPrototype
    v::V
    cutoff::C
    two_body_term::Bool
end
function TranscorrelatedHarmonicPotential(v; cutoff=nothing, two_body_term=false)
    return TranscorrelatedHarmonicPotential(v, cutoff, two_body_term)
end

function initialize(tcd::TranscorrelatedHarmonicPotential, ham)
    if basis(ham) ≡ MomentumSpace()
        address = starting_address(ham)
        M = num_modes(address)
        t = only(ham.t)
        u = only(ham.u)
        if ham isa Transcorrelated
            cutoff = isnothing(tcd.cutoff) ? ham.cutoff : tcd.cutoff
        else
            cutoff = isnothing(tcd.cutoff) ? 1 : tcd.cutoff
        end
        v = parameter_column(address, tcd.v)

        term = FullOneBodyTerm(TCDeltaFunctionOneBody(M, cutoff, v, t))
        if tcd.two_body_term && u ≠ 0
            return term + FullTwoBodyTerm(TCDeltaFunctionTwoBody(M, cutoff, v, u, t))
        else
            return term
        end
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `TranscorrelatedHarmonicPotential`"
        ))
    end
end
