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
import Rimu: num_offdiagonals, get_offdiagonal, diagonal_element, offdiagonals, starting_address, LOStructure

export ConstFunction
export column
export KineticEnergy, MomentumTransfer, HarmonicOscillatorMom, HubbardMom

include("abstract.jl")
include("column.jl")

include("sum.jl")
include("momentumtransfer.jl")
include("threebody.jl")
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
using Rimu.Hamiltonians: n_to_k, correlation_factor

struct WFunction{M}
    values::NTuple{M,Float64}
end
function WFunction(M, cutoff)
    return WFunction(Tuple(Rimu.Hamiltonians.w_function.(0:M-1, cutoff)))
end
(w::WFunction)(n) = w.values[abs(n) + 1]

struct TFunction{M}
    w::WFunction{M}
    cutoff::Int
    t::Float64
    v::Float64
end
function TFunction(M, cutoff, t, v)
    return TFunction(WFunction(M, cutoff), cutoff, float(t), float(v))
end
function (t_fun::TFunction{M})(p, q, k) where {M}
    t, v = t_fun.t, t_fun.v
    k_pi = n_to_k(k, M)
    pmq_pi = n_to_k(p - q, M)
    cor_k = correlation_factor(k, t_fun.cutoff, M)
    return v/M + 2v/M * (cor_k * k_pi - cor_k * pmq_pi) + 2v^2/t * t_fun.w(k)
end

struct QFunction{M}
    cutoff::Int
    t::Float64
    v::Float64
end
function QFunction(M, cutoff, t, v)
    return QFunction{M}(cutoff, float(t), float(v))
end
function (q_fun::QFunction{M})(k, l) where {M}
    t, v = q_fun.t, q_fun.v
    cor_k = correlation_factor(k, q_fun.cutoff, M)
    cor_l = correlation_factor(l, q_fun.cutoff, M)

    return -v^2/(t * M^2) * cor_k * cor_l
end

function Transcorrelated(address; t=1, v=1, v_ho=0, cutoff=1, three_body_term=true)
    M = num_modes(address)

    t_fun = TFunction(num_modes(address), cutoff, t, v)

    op = KineticEnergy(address, t; dispersion=continuum_dispersion) +
        MomentumTransfer(address, t_fun; fold=false)
    if three_body_term
        op += ThreeBodyMomentumTransfer(address, QFunction(M, cutoff, t, v))
    end
    if v_ho ≠ 0
        op += HarmonicOscillatorMom(address, v_ho)
    end
    return op
end

end # module
