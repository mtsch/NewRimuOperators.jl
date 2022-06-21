using Rimu.Hamiltonians: n_to_k, correlation_factor

struct QTensor
end

struct WTensor
end

get_dimension_from_address_somehow_probably_need_to_use_lattice_geometry(_) = 1

"""
    TranscorrelatedGeneral(address; kwargs...)

Implements a transcorrelated Hamiltonian any number of components with arbitrary masses 
in up to three spatial dimensions using momentum space formalism. Assumes that the 
interparticle interactions are contact interactions represented by Bethe-Peierls 
boundary conditions.

This is a generalisation of [Jeszenski *et al.* (2018)](https://arxiv.org/abs/1806.11268) 
and [Jeszenski *et al.* (2020)](https://arxiv.org/abs/2002.05987).

```math
\\begin{aligned}
\\tilde{H} - H &= 
\\end{aligned}
```

# Arguments

* `address`: The starting address, defines number of particles and sites.
* `u`: An array of interaction strengths between particles. Diagonal (offdiagonal) 
elements set intercomponent (intracomponent) interaction strengths. 
offdiagonal elements are intercomponent interaction strengths.
* `masses`: The masses of each component, in units of an arbitrary mass ratio ``m_0``.
* `BHJs`: Alternatively, set the Bose-Hubbard parameters `J` explicitly. Overrides `masses`
* `dim`: Set the dimension of the system. Defaults to 1D.
* `cutoff`: integer `n` that sets the cutoff momentum ``k_c = 2\\pi n / M``
* `threebodyQ`: flag to enable or disable three-body terms

# See also

* [`Transcorrelated1D`](@ref)
"""
struct TranscorrelatedGeneral{A,C,D,O<:AbstractOperator{Float64}} <: Hamiltonian{A,Float64}
    address::A
    u::SMatrix{C,C,Float64} # interparticle interaction parameters, inter- and intracomponent
    t::SVector{C,Float64} # energy scales; sets component masses
    cutoff::Int
    terms::O
    threebodyQ::Bool
end

function TranscorrelatedGeneral(
    add::CompositeFS{C}; 
    u::Union{Real,SMatrix{Real}}, 
    BHJs=nothing, 
    masses=ones(::Real,C), 
    cutoff::Int=1,
    threebodyQ::Bool=true
    ) where {C}

    M = num_modes(add)
    D = get_dimension_from_address_somehow_probably_need_to_use_lattice_geometry(address)

    if isnothing(j)
        t = float(M^2/2[masses...])
    else
        @assert length(BHJs) == C
        t = float(BHJs)
    end

    # bare hamiltonian
    ke = ParticleCountTerm(KineticEnergyFunction(add, t, continuum_dispersion))
    int = MomentumTwoBodyTerm(u/M)
    bare = ke + int

    # intra-component
    ke = ParticleCountTerm(KineticEnergyFunction(add, t, continuum_dispersion))
    non_herm = 0
    ddx_sq = 0
    inter = ke + non_herm + ddx_sq
    if threebodyQ
        inter += 0
    end

    # inter-component
    ke = ParticleCountTerm(KineticEnergyFunction(add, (t[a] + t[b])/2, continuum_dispersion))
    non_herm = 0
    ddx_sq = 0
    intra = ke + non_herm + ddx_sq

    # mixed-component
    mixed = 0

    # total
    terms = bare + inter + intra + mixed

    # three component
    if threebodyQ
        three = 0
        terms += three
    end

    return TranscorrelatedGeneral{typeof(add),C,D,typeof(terms)}(add,u,t,cutoff,terms,threebodyQ)
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