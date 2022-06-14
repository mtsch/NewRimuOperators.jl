struct Hubbard{A,B,O<:AbstractOperator{Float64}} <: Hamiltonian{A,Float64}
    address::A
    basis::B
    u
    t
    terms::O
end

function Hubbard(
    address;
    u::Union{AbstractMatrix{<:Real},Real}=1.0,
    t::Union{AbstractVector{<:Real},Real}=1.0,
    momentum=true,
    dispersion=hubbard_dispersion,
)
    basis = momentum ? MomentumSpace() : RealSpace()

    # Create appropriate terms
    if momentum
        M = num_modes(address)
        kinetic_term = ParticleCountTerm(KineticEnergyFunction(address, t, dispersion))
        interaction_term = MomentumTwoBodyTerm(interaction_matrix(address, u ./ M); fold=true)
    else
        if dispersion ≢ hubbard_dispersion
            throw(ArgumentError("setting dispersion is not supported in real space"))
        end
        kinetic_term = NeighbourOneBodyTerm(parameter_column(address, t))
        interaction_term = OnsiteInteractionTerm(interaction_matrix(address, u))
    end
    terms = kinetic_term + interaction_term

    return Hubbard(address, basis, u, t, terms)
end

function Base.show(io::IO, h::Hubbard)
    print(io, "Hubbard(")
    print(IOContext(io, :compact => true), h.address)
    print(io, ", u=$(h.u), t=$(h.t)")
    if basis(h) ≡ MomentumSpace()
        print(io, ", momentum=true)")
    else
        print(io, ", momentum=false)")
    end
end

starting_address(h::Hubbard) = h.address
LOStructure(::Hubbard) = IsHermitian()
terms(h::Hubbard) = h.terms
basis(h::Hubbard) = h.basis
