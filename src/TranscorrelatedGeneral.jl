using Rimu.Hamiltonians: n_to_k, correlation_factor

struct OneBodyJastrowFunction
end

struct TwoBodyJastrowFunction
end

struct TwoBodyNonHermitianFunction
end

struct OneBodyNonHermitianFunction
end

struct QTensor
end

struct WTensor
end

# Currently only implemented for 1D lattice
get_dimension_from_address_somehow_probably_need_to_use_lattice_geometry(_) = 1

"""
    TranscorrelatedGeneral1D(address; kwargs...)

Implements a transcorrelated Hamiltonian for any number of components with arbitrary masses 
on a lattice using momentum space formalism. Assumes that the interparticle interactions 
are contact interactions represented by Bethe-Peierls boundary conditions.

This is a generalisation of [Jeszenski *et al.* (2018)](https://arxiv.org/abs/1806.11268) 
and [Jeszenski *et al.* (2020)](https://arxiv.org/abs/2002.05987).

```math
\\begin{aligned}
\\tilde{H} - H &= 
\\end{aligned}
```

# Arguments

* `address`: The starting address, defines number of particles and sites.
* `u`: A symmetric matrix of interaction strengths between particles. Diagonal (offdiagonal) 
elements set intracomponent (intercomponent) interaction strengths. 
* `masses`: The masses of each component, in units of an arbitrary mass ratio ``m_0``.
* `BHJs`: Alternatively, set the Bose-Hubbard parameters `J` explicitly. Overrides `masses`.
* `dim`: Set the dimension of the system. Defaults to 1D.
* `cutoff`: integer `n` that sets the cutoff momentum ``k_c = 2\\pi n / M``
* `threebodyQ`: flag to enable or disable three-body terms

# See also

* [`Transcorrelated1D`](@ref)
"""
struct TranscorrelatedGeneral{A,C,D,O<:AbstractOperator{Float64}} <: Hamiltonian{A,Float64}
    address::A
    u::SMatrix{C,C,Float64} # interparticle interaction parameters, inter- and intracomponent
    j::SVector{C,Float64} # energy scales; sets component masses
    cutoff::Int
    terms::O
    threebody::Bool
end

function TranscorrelatedGeneral(add::CompositeFS{C}; 
    u = ones(C,C), 
    BHJs = nothing, 
    masses = ones(C), 
    cutoff = 1,
    threebody = false,
    fold = false
    ) where {C}

    M = num_modes(add)
    D = get_dimension_from_address_somehow_probably_need_to_use_lattice_geometry(address)

    if isnothing(BHJs)
        @assert length(masses) == C
        jdiag = float(masses .* M^2/2)
    else
        @assert length(BHJs) == C
        jdiag = float(BHJs)
    end
    j = Diagonal(jdiag)
    for a in 1:C, b in a+1:C
        j[a,b] = (j[a] + j[b])/2
        j[b,a] = j[a,b]
    end

    Q = QTensor(cutoff)
    W = WTensor(cutoff)

    # bare hamiltonian
    ke = ParticleCountTerm(KineticEnergyFunction(add, j, continuum_dispersion))
    int = MomentumTwoBodyTerm(u ./ M)
    pe = HarmonicPotential(v)
    bare = ke + int + pe

    # intra-component
    ke = MomentumTwoBodyTerm(KineticEnergyFunction(add, u/M, continuum_dispersion))
    non_herm = MomentumTwoBodyTerm(TwoBodyNonHermitianFunction(ones(C), cutoff); fold)
    ddx_sq = MomentumTwoBodyTerm(Wfunction(W, u.^2 ./ j / M^2); fold)
    intra = ke + non_herm + ddx_sq
    if threebody
        intra += MomentumThreeBodyTerm(Qfunction(Q, -diag(u).^2 ./ diag(j) / M^2))
    end

    # inter-component
    ke = ParticleCountTerm(KineticEnergyFunction(add, 2u/M, continuum_dispersion))
    non_herm = MomentumTwoBodyTerm(TwoBodyNonHermitianFunction(j, cutoff); fold)
    ddx_sq = MomentumTwoBodyTerm(Wfunction(u.^2 ./ j / M^2, cutoff); fold)
    inter = ke + non_herm + ddx_sq
    if threebody
        intra += MomentumThreeBodyTerm(Qfunction(Q, -u.^2 ./ j / M^2); fold)
    end

    # total
    terms = bare + intra + inter

    # mixed and three component
    if threebody
        mixed += MomentumThreeBodyTerm(; fold)
        three = MomentumThreeBodyTerm(; fold)
        terms += three - mixed
    end

    return TranscorrelatedGeneral{typeof(add),C,D,typeof(terms)}(add, u, j, cutoff, terms, threebody)
end

function Base.show(io::IO, h::TranscorrelatedGeneral)
    print(io, "TranscorrelatedGeneral(")
    print(IOContext(io, :compact => true), h.address)
    print(io, ", u=$(h.u), t=$(h.t), v=$(h.v), cutoff=$(h.cutoff), three_body_term=$(h.three_body_term)")
    print(io, ")")
end

starting_address(h::TranscorrelatedGeneral) = h.address
LOStructure(::Type{<:TranscorrelatedGeneral}) = AdjointKnown()
function Base.adjoint(h::TranscorrelatedGeneral)
    return TranscorrelatedGeneral(h.address, h.u, h.t, h.cutoff, adjoint(h.terms), h.three_body_term)
end
terms(h::TranscorrelatedGeneral) = h.terms
basis(h::TranscorrelatedGeneral) = h.basis