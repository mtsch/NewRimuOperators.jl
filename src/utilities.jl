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

struct ParameterColumnFunction{N,T}
    values::SVector{N,T}
end
(pcf::ParameterColumnFunction)(i, args...) = pcf.values[i]
