# TODO:

# terms.jl:
# How to handle adjoints? Most of the time, the terms will be Hermitian, but some of the
# times, they are not. In some cases it works well (e.g. `MomentumTwoBodyTerm`), but adding
# the option for all terms would be annoying at best.
# Perhaps the "functions" you pass to the terms could know whether permuting their arguments
# changes anything - this way the terms would know what structure they have.

# sum.jl:
# Idea: summing should merge like terms (e.g. summing a harmonic potential with a delta
# potential should be able to merge the `FullOneBodyTerm`s.

# abstract.jl:
# Idea: perhaps the type hirearchy should be reworked a bit. These terms don't really need
# to know the starting address. Hamiltonians could still be required to provide them.
# A potential (or more general Hamiltonian addon) could inherit the starting address from
# the Hamiltonian it's being added to.

# abstract.jl:
# Idea: add IsDiagonal to LOStructure. Then they can be skipped when generating
# offdiagonals. Another option would be to add IsSymmetric, which would provide an adjoint
# implementation.


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
export Hubbard, Transcorrelated
export HarmonicPotential, DeltaPotential, TranscorrelatedDeltaPotential

include("utilities.jl")
include("abstract.jl")
include("column.jl")
include("sum.jl")
include("terms.jl")
include("extensions.jl")

include("hubbard.jl")
include("transcorrelated.jl")

include("potentials.jl")

end # module
