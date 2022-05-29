struct Hubbard{A,B,N,O<:AbstractOperator{Float64}} <: Hamiltonian{A,Float64}
    address::A
    basis::B
    u::SMatrix{N,N,Float64}
    t::SVector{N,Float64}
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

    C = num_components(address)
    M = num_modes(address)
    # TODO here
    if u isa AbstractMatrix
        # Check that matrix is well-formed
        size(u) == (C, C) || throw(ArgumentError("`u` must be a $C×$C matrix or a scalar"))
        issymmetric(u) || throw(ArgumentError("`u` must be symmetric"))
        Rimu.Hamiltonians.warn_fermi_interaction(address, u)
        us = SMatrix{C,C,Float64}(u)
    else
        us = SMatrix{C,C,Float64}(fill(u, (C, C)))
    end
    # Create appropriate terms
    if momentum
        kinetic_term = ParticleCountTerm(KineticEnergyFunction(address, t, dispersion))
        interaction_term = MomentumTwoBodyTerm(InteractionMatrixFunction(us./M); fold=true)
    else
        if dispersion ≢ hubbard_dispersion
            throw(ArgumentError("setting dispersion is not supported in real space"))
        end
        kinetic_term = NeighbourOneBodyTerm(ParameterColumnFunction(address, t))
        interaction_term = OnsiteInteractionTerm(InteractionMatrixFunction(us))
    end
    terms = kinetic_term + interaction_term

    return Hubbard(address, basis, us, to_parameter_vector(address, t), terms)
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
