module NewRimuOperators

using StaticArrays
using LinearAlgebra

using Rimu
# Import some Rimu internals:
using Rimu.Hamiltonians: momentum_transfer_excitation
using Rimu.Hamiltonians: momentum_transfer_diagonal
using Rimu.Hamiltonians: momentum_external_potential_excitation
using Rimu.Hamiltonians: momentum_external_potential_diagonal
using Rimu.Hamiltonians: momentum_space_harmonic_potential
using Rimu.Hamiltonians: AbstractOffdiagonals
using Rimu.BitStringAddresses: update_component
# These will be extended
import Rimu: num_offdiagonals, get_offdiagonal, diagonal_element, offdiagonals, starting_address, LOStructure

export ConstFunction, InteractionMatrix
export column
export HarmonicOscillatorMom, HarmonicOscillatorReal
export HubbardMom, HubbardReal, Transcorrelated

include("utilities.jl")
include("abstract.jl")
include("column.jl")
include("sum.jl")

include("momentumtransfer.jl")
include("threebody.jl")
include("kineticenergy.jl")
include("harmonicoscillator.jl")

include("onsiteinteraction.jl")
include("realspacehop.jl")

include("hubbard.jl")
include("transcorrelated.jl")

end # module
