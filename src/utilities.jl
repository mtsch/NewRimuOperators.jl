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

Function that returns `x` for all arguments. Equivalent to `(args...) -> x`.
"""
struct ConstFunction{T}
    value::T
end
(c::ConstFunction)(args...; kwargs...) = c.value

function to_parameter_vector(address, t::Number)
    C = num_components(address)
    return SVector{C}(fill(float(t), C))
end
function to_parameter_vector(address, t)
    C = num_components(address)
    length(t) == C || throw(ArgumentError("`t` must be a $C-vector or a scalar"))
    return SVector{C}(float.(t))
end

"""
     parameter_column(add, t)

Create a parameter column and check that it matches the address `add`. Returns
`ConstFunction` if length of `t` is 1 and [`ParameterColumnFunction`](@ref) otherwise.
"""
function parameter_column(add, t)
    C = num_components(add)
    if length(t) == 1
        return ConstFunction(first(t))
    elseif C == length(t)
        return ParameterColumnFunction(SVector{C}(t))
    else
        throw(ArgumentError(
            "Length of `$t` does not match number of components in address"
        ))
    end
end

"""
    ParameterColumnFunction(values::SVector)

Wrapper over `SVector` that turns function calls into `getindex` calls, i.e. `pf(i) =
pf.values[i]`.
"""
struct ParameterColumnFunction{N,T}
    values::SVector{N,T}
end
(pcf::ParameterColumnFunction)(i, args...) = pcf.values[i]

struct KineticEnergyFunction{M,T}
    t::T
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
    return KineticEnergyFunction(parameter_column(address, t), kes)
end
(ke::KineticEnergyFunction)(σ, p) = ke.kes[p] * ke.t(σ)


function interaction_matrix(add, u)
    C = num_components(add)
    if length(u) == 1
        return ConstFunction(first(u))
    elseif length(u) == C * C
        matrix = SMatrix{C, C}(u)
        issymmetric(matrix) || throw(ArgumentError("interaction matrix must be symmetric"))
        Rimu.Hamiltonians.warn_fermi_interaction(add, matrix)
        return InteractionMatrixFunction(matrix)
    else
        throw(ArgumentError(
            "Length of `$u` does not match number of components in address"
        ))
    end
end

"""
    InteractionMatrixFunction(SMatrix)

Wrapper over square `SMatrix` that turns function calls into `getindex` calls, i.e. `imf(i,
j) = imf.values[i, j]`.
"""
struct InteractionMatrixFunction{N,T,M<:SMatrix{N,N,T}}
    values::M
end
(imf::InteractionMatrixFunction)(i, j, args...) = imf.values[i, j]
