"""
    abstract type AbstractOperator{A,T}

Supertype for linear operators.

# Interface

* [`diagonal_element`](@ref)
* [`get_offdiagonal`](@ref)
* [`num_offdiagonals`](@ref)
* [`column`](@ref) (optional)
"""
# TODO: the subtyping needs to be the other way around -- a Hamiltonian is an operator
abstract type AbstractOperator{T} <: AbstractHamiltonian{T} end

# Rimu compatibility
offdiagonals(op::AbstractOperator, add) = offdiagonals(column(op, add))
get_offdiagonal(op::AbstractOperator, add, i) = offdiagonals(column(op, add))[i]
num_offdiagonals(op::AbstractOperator, add) = length(offdiagonals(column(op, add)))
diagonal_element(op::AbstractOperator, add) = diagonal_element(column(op, add))

"""
    abstract type Hamiltonian{A,T} <: AbstractOperator{A,T}

A subtype of `Hamitlonian` may define `terms` which should return an operator, probably
a sum of the terms that appear in the Hamiltonian.

This is mainly done to make the user experience more pleasant.

TODO: rename me to `AbstractHamiltonian` later.
"""
abstract type Hamiltonian{A,T} <: AbstractOperator{T} end

diagonal_element(h::Hamiltonian, args...) = diagonal_element(terms(h), args...)
num_offdiagonals(h::Hamiltonian, args...) = num_offdiagonals(terms(h), args...)
get_offdiagonal(h::Hamiltonian, args...) = get_offdiagonal(terms(h), args...)
column(h::Hamiltonian{A}, add::A) where {A} = column(terms(h), add)
column(h::Hamiltonian{A}, add::A) where {A<:CompositeFS} = column(terms(h), add)

is_real_space(::Hamiltonian) = false
is_mom_space(::Hamiltonian) = false

"""

TODO document me!
"""
abstract type Basis end

struct RealSpace end
struct MomentumSpace end

"""
    abstract type AbstractColumn{A,T}

Packs together diagonal and offdiagonal elements of an operator at selected address.

See [`Column`](@ref) for a default implementation.

# Interface

* [`diagonal_element`](@ref)
* [`get_offdiagonal`](@ref)
* [`num_offdiagonals`](@ref)
* [`offdiagonals`](@ref) (optional)
"""
abstract type AbstractColumn{A,T} end

offdiagonals(col::AbstractColumn) = ColOffdiagonals(col)

"""
    Column <: AbstractColumn

Default implementation of [`AbstractColumn`](@ref). Defined in terms of
[`diagonal_element`](@ref), [`get_offdiagonal`](@ref), and [`num_offdiagonals`](@ref)
of the operator.
"""
struct Column{A,T,O<:AbstractOperator{T},M} <: AbstractColumn{A,T}
    operator::O
    address::A
    map::M
end
diagonal_element(col::Column) = diagonal_element(col.operator, col.address, col.map)
get_offdiagonal(col::Column, i) = get_offdiagonal(col.operator, col.address, col.map, i)
num_offdiagonals(col::Column) = num_offdiagonals(col.operator, col.address, col.map)

Base.show(io::IO, col::Column) = print(io, "column(", col.operator, ", ", col.address, ")")

function column(op::AbstractOperator, add::A) where {A}
    map = OccupiedModeMap(add)
    return Column(op, add, map)
end

struct ColOffdiagonals{A,T,C<:AbstractColumn{A,T}} <: AbstractVector{Tuple{A,T}}
    column::C
end
Base.size(od::ColOffdiagonals) = (num_offdiagonals(od.column),)
Base.getindex(od::ColOffdiagonals, i) = get_offdiagonal(od.column, i)
