"""
    ConstFunction(x)

Construct a function that returns `x` for all arguments.
"""
struct ConstFunction{T}
    value::T
end
(c::ConstFunction)(args...; kwargs...) = c.value

"""
    abstract type AbstractOperator{A,T}

Supertype for linear operators.

# Interface

* [`diagonal_element`](@ref)
* [`get_offdiagonal`](@ref)
* [`num_offdiagonals`](@ref)
* [`offdiagonals`](@ref) (optional)
"""
# TODO: the subtyping needs to be the other way around -- a Hamiltonian is an operator
abstract type AbstractOperator{A,T} <: AbstractHamiltonian{T} end

# TODO: how to name this?
"""
    CompositeAction(op)

Trait that determines how an operator acts across components with a [`CompositeFS`](@ref).

# Options

* [`OneWayCompositeAction`](@ref)
* [`TwoWayCompositeAction`](@ref)
* [`NoCompositeAction`](@ref)
"""
abstract type CompositeAction end
"""
    OneWayCompositeAction

Operator acts symmetrically across components.

In other words, `excitation(op, add_a, add_b)` also performs `excitation(op, add_b, add_a)`
implicitly.
"""
struct OneWayCompositeAction <: CompositeAction end
"""
    TwoWayCompositeAction

Operator acts differently across components depending on direction.

In other words, both `excitation(op, add_a, add_b)` and `excitation(op, add_b, add_a)` must
be called separately to get the full effect.
"""
struct TwoWayCompositeAction <: CompositeAction end

"""
    NoCompositeAction

Operator does not act across components.

In other words, `excitation(op, add_a, add_b) = 0`.
"""
struct NoCompositeAction <: CompositeAction end

# Rimu compatibility
offdiagonals(op::AbstractOperator, add) = offdiagonals(column(op, add))
get_offdiagonal(op::AbstractOperator, add, i) = offdiagonals(column(op, add))[i]
num_offdiagonals(op::AbstractOperator, add) = length(offdiagonals(column(op, add)))
diagonal_element(op::AbstractOperator, add) = diagonal_element(column(op, add))
