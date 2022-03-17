struct HarmonicOscillatorMom{A,M} <: AbstractOperator{A,Float64}
    address::A
    potential::SVector{M,Float64}
end
function HarmonicOscillatorMom(address, v)
    M = num_modes(address)
    return HarmonicOscillatorMom(address, momentum_space_harmonic_potential(M, v))
end

starting_address(op::HarmonicOscillatorMom) = op.address

num_offdiagonals(op::HarmonicOscillatorMom, add, map) = length(map) * (num_modes(add) - 1)

function get_offdiagonal(op::HarmonicOscillatorMom, add, map, i)
    new_add, value = momentum_external_potential_excitation(op.potential, add, i, map)
    return new_add, value
end
function diagonal_element(op::HarmonicOscillatorMom, add, map)
    return momentum_external_potential_diagonal(op.potential, add, map)
end

CompositeAction(::HarmonicOscillatorMom) = NoCompositeAction()
LOStructure(::HarmonicOscillatorMom) = IsHermitian()
