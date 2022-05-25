import Rimu.BitStringAddresses: update_component
function update_component(fs::CompositeFS{2}, new_add_a, new_add_b, ::Val{1}, ::Val{2})
    return typeof(fs)(new_add_a, new_add_b)
end
function update_component(fs::CompositeFS{2}, new_add_a, new_add_b, ::Val{2}, ::Val{1})
    return typeof(fs)(new_add_b, new_add_a)
end
function update_component(
    fs::CompositeFS, new_add_a, new_add_b, ::Val{A}, ::Val{B}
) where {A,B}
    fs = update_component(fs, new_add_a, Val(A))
    return update_component(fs, new_add_b, Val(B))
end


"""
    ConstFunction(x)

Construct a function that returns `x` for all arguments.
"""
struct ConstFunction{T}
    value::T
end
(c::ConstFunction)(args...; kwargs...) = c.value

struct InteractionMatrixFunction{N,T,M<:SMatrix{N,N,T}}
    values::M
end
(imf::InteractionMatrixFunction)(i, j, args...) = imf.values[i, j]

function to_parameter_vector(address, t::Number)
    C = num_components(address)
    return SVector{C}(fill(float(t), C))
end
function to_parameter_vector(address, t)
    C = num_components(address)
    length(t) == C || throw(ArgumentError("`t` must be a $C-vector or a scalar"))
    return SVector{C}(float.(t))
end

struct ParameterColumnFunction{N,T}
    values::SVector{N,T}
end
function ParameterColumnFunction(address, t)
    return ParameterColumnFunction(to_parameter_vector(address, t))
end
(pcf::ParameterColumnFunction)(i, args...) = pcf.values[i]

struct KineticEnergyFunction{K,M}
    t::SVector{K,Float64}
    kes::SVector{M,Float64}
end

function KineticEnergyFunction(address, t, dispersion)
    M = num_modes(address)
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step=step, length=M)
    ks = SVector{M}(kr)
    kes = SVector{M}(dispersion.(kr))
    return KineticEnergyFunction(to_parameter_vector(address, t), kes)
end
(ke::KineticEnergyFunction)(σ, p) = ke.kes[p] * ke.t[σ]
