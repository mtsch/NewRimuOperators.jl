module NewRimuOperators

using StaticArrays

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
import Rimu: num_offdiagonals, get_offdiagonal, diagonal_element, offdiagonals, starting_address

export ConstFunction
export column
export KineticEnergy, MomentumTransfer, HarmonicOscillatorMom, HubbardMom

include("abstract.jl")
include("column.jl")

include("sum.jl")
include("momentumtransfer.jl")
include("dispersion.jl")
include("harmonicoscillator.jl")

# Below are some examples on how one would go about writing a Hamiltonian:
function HubbardMom(address; t=1, u=1, v_ho=0)
    M = num_modes(address)
    op = KineticEnergy(address, t) + MomentumTransfer(address, u/2M)
    if v_ho ≠ 0
        op += HarmonicOscillatorMom(address, v_ho)
    end
    return op
end

# Ignore below, not implemented.
struct WFunction{M}
    values::NTuple{M,Float64}
end
function WFunction(m, cutoff)
    return WFunction(Tuple(Rimu.Hamiltonians.w_function.(0:M-1, cutoff)))
end
(w::WFunction)(n) = w.values[abs(n) + 1]

struct TFunction{M}
    w::WFunction{M}
    t::Float64
    v::Float64
end
function TFunction(m, cutoff, t, v)
    return TFunction(WFunction(m, cutoff), t, v)
end
function (t_fun::TFunction{M})(p, q, k) where {M}
    t, v = t_fun.t, t_fun.v
    k_pi = n_to_k(k, M)
    pmq_pi = n_to_k(p - q, M)
    cor_k = correlation_factor(h, k)
    return v/M + 2v/M * (cor_k * k_pi - cor_k * pmq_pi) + 2v^2/t * t_fun.w(k)
end

function Transcorrelated(address; t=1, v=1, v_ho=0, cutoff=1)
    M = num_modes(address)

    t_fun = TFunction()

    op = Dispersion(address, t) + MomentumTransfer(address, t_fun)
    # 3-body here
    if v_ho ≠ 0
        op += HarmonicOscillatorMom(address, v_ho)
    end
    return op
end

end # module
