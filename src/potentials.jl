using Rimu.Hamiltonians: i_to_k, shift_lattice

"""
    MomPotentialFunction(v, potential)

Function that when combined with [`FullOneBodyTerm`](@ref) acts as a potential in momentum
space.

# Arguments

* `v`: a function or callable object that returns the strength of the potential for given
  component.
* `potential`: a `SVector` of potential values. The `potential` passed to the constructor
  must already be Fourier transformed!
"""
struct MomPotentialFunction{M,V}
    v::V
    potential::SVector{M,Float64}
end
function (f::MomPotentialFunction{M})(σ, p, q) where {M}
    return f.v(σ) * f.potential[mod(p - q, M) + 1]
end

"""
    RealPotentialFunction(v, potential)

Function that when combined with [`ParticleCountTerm`](@ref) acts as a potential in real
space.

# Arguments

* `v`: a function or callable object that returns the strength of the potential for given
  component.
* `potential`: a `SVector` of potential values.
"""
struct RealPotentialFunction{M,V}
    v::V
    potential::SVector{M,Float64}
end
function (f::RealPotentialFunction)(σ, i)
    return f.v(σ) * f.potential[i]
end

"""
    HarmonicPotential(v_ho)

Harmonic potential with strength `v_ho`. The argument `v_ho` can be a real number or a
vector of real numbers whose length matches the number of components in the Hamiltonian.

Can be used with both real and momentum space Hamiltonians.
"""
struct HarmonicPotential{V} <: ExtensionPrototype
    v_ho::V
end

function initialize(hp::HarmonicPotential, ham)
    address = starting_address(ham)
    M = num_modes(address)
    v_ho = ParameterColumnFunction(address, hp.v_ho)

    if basis(ham) ≡ MomentumSpace()
        fun = MomPotentialFunction(v_ho, momentum_space_harmonic_potential(M, 1))
        return FullOneBodyTerm(fun)
    elseif basis(ham) ≡ RealSpace()
        is = range(-fld(M, 2); length=M) # [-M÷2, M÷2) including left boundary
        js = shift_lattice(is) # shifted such that js[1] = 0
        potential = SVector{M}(float.(js).^2)
        fun = RealPotentialFunction(v_ho, potential)
        return ParticleCountTerm(fun)
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `HarmonicPotential`"
        ))
    end
end

"""
    DeltaPotential(v)

Delta function potential with strength `v`. The argument `v` can be a real number or a
vector of real numbers whose length matches the number of components in the Hamiltonian.

Can be used with both real and momentum space Hamiltonians.
"""
struct DeltaPotential{V} <: ExtensionPrototype
    v::V
end

function initialize(dp::DeltaPotential, ham)
    address = starting_address(ham)
    M = num_modes(address)
    v = to_parameter_vector(address, dp.v)
    if basis(ham) ≡ MomentumSpace()
        term = FullOneBodyTerm(ParameterColumnFunction(address, v ./ M))
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `DeltaPotential`"
        ))
    end
end

"""
    TCDeltaFunctionOneBody(M, cutoff, v, t)

The function used in the one-body term of the transcorrelated delta function potential.
"""
struct TCDeltaFunctionOneBody{M,V}
    corr::CorrelationFactor{M}
    w::WFunction{M}
    v::V
    t::Float64
end

function TCDeltaFunctionOneBody(M::Int, cutoff::Int, v, t)
    corr = CorrelationFactor(M, cutoff)
    w = WFunction(M, cutoff)
    return TCDeltaFunctionOneBody(corr, w, v, t)
end

function (d::TCDeltaFunctionOneBody{M})(σ, p, q) where {M}
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
    TCDeltaFunctionTwoBody(M, cutoff, v, t)

The function used in the two-body term of the transcorrelated delta function potential.
"""
struct TCDeltaFunctionTwoBody{M,V}
    corr::CorrelationFactor{M}
    v::V
    u::Float64
    t::Float64
end
function TCDeltaFunctionTwoBody(M, cutoff, v::V, u, t) where {V}
    TCDeltaFunctionTwoBody(CorrelationFactor(M, cutoff; length=2M), v, u, t)
end

function (d::TCDeltaFunctionTwoBody{M})(σ, τ, s, r, q, p) where {M}
    u = d.u; v = d.v; t = d.t; corr = d.corr
    result_στ = (v(σ) * corr(p + q - r - s) * corr(r - q))
    result_τσ = (v(τ) * corr(p + q - r - s) * corr(s - p))
    return 16u/(t*M^2) * (result_στ + result_τσ)
end

"""
    TranscorrelatedDeltaPotential(v; cutoff, two_body_term=false)

Transcorrelated delta function potential with strength `v`. The argument `v` can be a real
number or a vector of real numbers whose length matches the number of components in the
Hamiltonian.

The `cutoff` defaults to the cutoff of the Hamiltonian is attached to, or 1.

Can only be used with momentum space Hamiltonians.
"""
struct TranscorrelatedDeltaPotential{V,C} <: ExtensionPrototype
    v::V
    cutoff::C
    two_body_term::Bool
end
function TranscorrelatedDeltaPotential(v; cutoff=nothing, two_body_term=false)
    return TranscorrelatedDeltaPotential(v, cutoff, two_body_term)
end

function initialize(tcd::TranscorrelatedDeltaPotential, ham)
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
        v = ParameterColumnFunction(address, tcd.v)

        term = FullOneBodyTerm(TCDeltaFunctionOneBody(M, cutoff, v, t))
        if tcd.two_body_term
            return term + FullTwoBodyTerm(TCDeltaFunctionTwoBody(M, cutoff, v, u, t))
        else
            return term
        end
    else
        throw(ArgumentError(
            "Basis `$(basis(ham))` is not compatible with `TranscorrelatedDeltaPotential`"
        ))
    end
end
