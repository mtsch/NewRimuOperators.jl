struct KineticEnergyFunction{K,M}
    t::SVector{K,Float64}
    kes::SVector{M,Float64}
end

function KineticEnergyFunction(address, t, dispersion)
    M = num_modes(address)
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step=step, length=M)
    ks = SVector{M}(kr)
    kes = SVector{M}(dispersion.(kr))
    return KineticEnergyFunction(t, kes)
end
(ke::KineticEnergyFunction)(σ, p) = ke.kes[p] * ke.t[σ]

struct Hubbard{A,N,K,I} <: Hamiltonian{A,Float64}
    address::A
    u::SMatrix{N,N,Float64}
    t::SVector{N,Float64}
    kinetic_term::K
    interaction_term::I
end

function Hubbard(
    address;
    u::Union{AbstractMatrix{<:Real},Real}=1.0,
    t::Union{AbstractVector{<:Real},Real}=1.0,
    momentum=true,
    dispersion=hubbard_dispersion,
)
    C = num_components(address)
    M = num_modes(address)
    if u isa AbstractMatrix
        # Check that matrix is well-formed
        size(u) == (C, C) || throw(ArgumentError("`u` must be a $C×$C matrix or a scalar"))
        issymmetric(u) || throw(ArgumentError("`u` must be symmetric"))
        Rimu.Hamiltonians.warn_fermi_interaction(address, u)
        us = SMatrix{C,C,Float64}(u)
    else
        us = SMatrix{C,C,Float64}(fill(u, (C, C)))
    end
    if t isa AbstractArray
        length(t) == C || throw(ArgumentError("`t` must be a $C-vector or a scalar"))
        ts = SVector{C,Float64}(t)
    else
        ts = @SVector fill(Float64(t), C)
    end

    # Create appropriate terms
    if momentum
        kinetic_term = ParticleCountTerm(
            address, KineticEnergyFunction(address, ts, dispersion)
        )
        interaction_term = MomentumTwoBodyTerm(
            address, InteractionMatrixFunction(us ./ M); fold=true
        )
    else
        if dispersion ≢ hubbard_dispersion
            throw(ArgumentError("setting dispersion is not supported in real space"))
        end
        kinetic_term = NeighbourOneBodyTerm(address, ParameterColumnFunction(ts))
        interaction_term = OnsiteInteractionTerm(address, InteractionMatrixFunction(us))
    end

    return Hubbard(address, us, ts, kinetic_term, interaction_term)
end

function Base.show(io::IO, h::Hubbard)
    print(io, "Hubbard(")
    print(IOContext(io, :compact => true), h.address)
    print(io, ", u=$(h.u), t=$(h.t)")
    if h.kinetic_term isa ParticleCountTerm
        print(io, ", momentum=true)")
    else
        print(io, ", momentum=false)")
    end
end

starting_address(h::Hubbard) = h.address
LOStructure(::Hubbard) = IsHermitian()
terms(h::Hubbard) = h.kinetic_term + h.interaction_term
