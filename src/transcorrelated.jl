using Rimu.Hamiltonians: n_to_k, correlation_factor

struct WFunction{M}
    values::NTuple{M,Float64}
end
function WFunction(M, cutoff)
    return WFunction(Tuple(Rimu.Hamiltonians.w_function.(0:M-1, cutoff)))
end
(w::WFunction)(n) = w.values[abs(n) + 1]

struct CorrelationFactor{M}
    values::NTuple{M,Float64}
end
function CorrelationFactor(M, cutoff; length=M)
    CorrelationFactor(Tuple(correlation_factor.(1:length, cutoff, M)))
end
function (correlation_factor::CorrelationFactor{M})(n) where {M}
    absn = abs(n)
    if 0 < absn
        return sign(n) * correlation_factor.values[absn]
    else
        return 0.0
    end
end

struct TFunction{M}
    cutoff::Int
    t::Float64
    u::Float64
    correlation_factor::CorrelationFactor{M}
    w::WFunction{M}
end
function TFunction(M, cutoff, t, u)
    w = WFunction(M, cutoff)
    correlation_factor = CorrelationFactor(M, cutoff)
    return TFunction(cutoff, float(t), float(u), correlation_factor, w)
end
@fastmath function (t_fun::TFunction{M})(_, _, p, q, k) where {M}
    t, u = t_fun.t, t_fun.u
    k_pi = n_to_k(k, M)
    pmq_pi = n_to_k(p - q, M)
    cor_k = t_fun.correlation_factor(k)
    return u/M + 2u/M * (cor_k * k_pi - cor_k * pmq_pi) + 2u^2/t * t_fun.w(k)
end

struct QFunction{M}
    cutoff::Int
    t::Float64
    u::Float64
    correlation_factor::CorrelationFactor{M}
end
function QFunction(M, cutoff, t, u)
    return QFunction{M}(cutoff, float(t), float(u), CorrelationFactor(M, cutoff))
end
function (q_fun::QFunction{M})(_, _, k, l) where {M}
    t, u = q_fun.t, q_fun.u
    cor_k = q_fun.correlation_factor(k)
    cor_l = q_fun.correlation_factor(l)

    return -u^2/(t * M^2) * cor_k * cor_l
end

struct Transcorrelated{A,K,I} <: Hamiltonian{A,Float64}
    address::A
    t::Float64
    u::Float64
    cutoff::Int
    three_body_term::Bool
    kinetic_term::K
    interaction_term::I
end

function Transcorrelated(address; t::Real=1, u::Real=1, cutoff=1, three_body_term=true)
    M = num_modes(address)
    C = num_components(address)

    t_fun = TFunction(num_modes(address), cutoff, t, u)

    ts = @SVector fill(float(t), C)
    kinetic_term = ParticleCountTerm(
        address, KineticEnergyFunction(address, ts, continuum_dispersion)
    )
    interaction_term = MomentumTwoBodyTerm(address, t_fun; fold=false)
    if three_body_term
        interaction_term += MomentumThreeBodyTerm(address, QFunction(M, cutoff, t, u))
    end
    return Transcorrelated(
        address, Float64(t), Float64(u), cutoff, three_body_term,
        kinetic_term, interaction_term,
    )
end

function Base.show(io::IO, h::Transcorrelated)
    print(io, "Transcorrelated(")
    print(IOContext(io, :compact => true), h.address)
    print(io, "; t=$(h.t), u=$(h.u), cutoff=$(h.cutoff), three_body_term=$(h.three_body_term))")
end

terms(h::Transcorrelated) = h.interaction_term + h.kinetic_term

starting_address(h::Transcorrelated) = h.address
LOStructure(::Transcorrelated) = AdjointKnown()
function Base.adjoint(h::Transcorrelated)
    return Transcorrelated(
        h.address, h.t, h.u, h.cutoff,
        h.three_body_term, adjoint(h.interaction_term), h.kinetic_term
    )
end

is_mom_space(::Transcorrelated) = true
