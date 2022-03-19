"""
    ConstFunction(x)

Construct a function that returns `x` for all arguments.
"""
struct ConstFunction{T}
    value::T
end
(c::ConstFunction)(args...; kwargs...) = c.value

struct InteractionMatrixFunction{N,T}
    values::SMatrix{N,N,T}
end
(imf::InteractionMatrixFunction)(i, j, args...) = imf.values[i, j]

struct ParameterColumnFunction{N,T}
    values::SVector{N,T}
end
(pcf::ParameterColumnFunction)(i, args...) = pcf.values[i]
