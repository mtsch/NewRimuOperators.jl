# NOTE: if the address knew its basis, HubbardMom and HubbardReal could even be merged into
# a single struct by setting appropriate values into the last two fields.
struct HubbardMom{A,N,K<:KineticEnergy,M<:MomentumTransfer} <: Hamiltonian{A,Float64}
    address::A
    u::SMatrix{N,N,Float64}
    t::SVector{N,Float64}
    kinetic_energy::K
    momentum_transfer::M
end

function HubbardMom(address; dispersion=hubbard_dispersion, u=nothing, t=nothing)
    N = num_components(address)
    M = num_modes(address)
    if isnothing(u)
        u = ones(N,N)
    end
    if isnothing(t)
        t = ones(N)
    end

    us = SMatrix{N,N,Float64}(u)
    ts = SVector{N,Float64}(t)
    issymmetric(us) || throw(ArgumentError("`u` must be a symmetric matrix"))

    kinetic_energy = KineticEnergy(address, ParameterColumnFunction(ts); dispersion)
    mom_transfer = MomentumTransfer(address, InteractionMatrixFunction(us ./ M); fold=true)

    return HubbardMom(address, us, ts, kinetic_energy, mom_transfer)
end

function Base.show(io::IO, h::HubbardMom)
    print(io, "HubbardMom(")
    print(IOContext(io, :compact => true), h.address)
    print(io, ", u=$(h.u), t=$(h.t))")
end

starting_address(h::HubbardMom) = h.address
LOStructure(::HubbardMom) = IsHermitian()
components(h::HubbardMom) = h.kinetic_energy + h.momentum_transfer

struct HubbardReal{A,N,H<:RealSpaceHop,O<:OnsiteInteraction} <: Hamiltonian{A,Float64}
    address::A
    u::SMatrix{N,N,Float64}
    t::SVector{N,Float64}
    hopping::H
    interaction::O
end

function HubbardReal(address; dispersion=hubbard_dispersion, u=nothing, t=nothing)
    N = num_components(address)
    M = num_modes(address)
    if isnothing(u)
        u = ones(N,N)
    end
    if isnothing(t)
        t = ones(N)
    end

    us = SMatrix{N,N,Float64}(u)
    ts = SVector{N,Float64}(t)
    issymmetric(us) || throw(ArgumentError("`u` must be a symmetric matrix"))

    hopping = RealSpaceHop(address, ParameterColumnFunction(ts))
    interaction = OnsiteInteraction(address, InteractionMatrixFunction(us))

    return HubbardReal(address, us, ts, hopping, interaction)
end

function Base.show(io::IO, h::HubbardReal)
    print(io, "HubbardReal(")
    print(IOContext(io, :compact => true), h.address)
    print(io, ", u=$(h.u), t=$(h.t))")
end

starting_address(h::HubbardReal) = h.address
LOStructure(::HubbardReal) = IsHermitian()
components(h::HubbardReal) = h.hopping + h.interaction
