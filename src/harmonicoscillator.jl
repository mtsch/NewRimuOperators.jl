# NOTE: these harmonic oscillators should probably be replaced with general potentials
using Rimu.Hamiltonians: shift_lattice

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

# NOTE: this is basically the same thing as kinetic_energy, with a different
# constructor. Merge? Under what name?
struct HarmonicOscillatorReal{A,M} <: AbstractOperator{A,Float64}
    address::A
    potential::SVector{M,Float64}
end
function HarmonicOscillatorReal(address, v)
    M = num_modes(address)
    is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
    js = shift_lattice(is) # shifted such that js[1] = 0
    potential = SVector{M}(v*j^2 for j in js)
    return HarmonicOscillatorReal(address, potential)
end

starting_address(op::HarmonicOscillatorReal) = op.address
num_offdiagonals(op::HarmonicOscillatorReal, _, _) = 0

function diagonal_element(op::HarmonicOscillatorReal, _, map)
    result = 0.0
    for index in map
        result += index.occnum * op.potential[index.mode]
    end
    return result
end

CompositeAction(::HarmonicOscillatorReal) = NoCompositeAction()
LOStructure(::HarmonicOscillatorReal) = IsHermitian()
