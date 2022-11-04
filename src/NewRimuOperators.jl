# TODO:

# terms.jl: [idea in progress]
# How to handle adjoints? Most of the time, the terms will be Hermitian, but some of the
# times, they are not. In some cases it works well (e.g. `MomentumTwoBodyTerm`), but adding
# the option for all terms would be annoying at best.
# Perhaps the "functions" you pass to the terms could know whether permuting their arguments
# changes anything - this way the terms would know what structure they have.

# Add a function "modifier" that makes it adjoint. Could probably use a standard way to
# write the functions (one per N-body excitation).

# sum.jl: [in progress]
# Idea: summing should merge like terms (e.g. summing a harmonic potential with a delta
# potential should be able to merge the `FullOneBodyTerm`s.

# Having values in SVectors in ParameterColumnFunction et al hurts performance even if
# length is 1. This is not too bad, but it would be nice to avoid.


module NewRimuOperators

using StaticArrays
using LinearAlgebra
using FFTW

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

export InteractionMatrix
export column
export SingleTermOperator
export Hubbard, Transcorrelated
export TranscorrelatedPotential
export HarmonicPotential, DeltaPotential, TranscorrelatedDeltaPotential

include("utilities.jl")
include("abstract.jl")
include("terms.jl")
include("termcolumns.jl")

include("excitationhelpers.jl")
include("onebodyterms.jl")
include("twobodyterms.jl")
include("threebodyterms.jl")

export StupidTwoBodyTerm, StupidThreeBodyTerm
include("stupidterms.jl")

include("sum.jl")
include("extensions.jl")

include("hubbard.jl")
include("transcorrelated.jl")
include("potentials.jl")
include("TranscorrelatedPotential.jl")


end # module
