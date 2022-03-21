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
    Column

Default implementation of [`AbstractColumn`](@ref). Defined in terms of
[`diagonal_element`](@ref), [`get_offdiagonal`](@ref), and [`num_offdiagonals`](@ref)
of the operator.
"""
struct Column{A,T,O<:AbstractOperator{A,T},M} <: AbstractColumn{A,T}
    operator::O
    address::A
    map::M
end
diagonal_element(col::Column) = diagonal_element(col.operator, col.address, col.map)
get_offdiagonal(col::Column, i) = get_offdiagonal(col.operator, col.address, col.map, i)
num_offdiagonals(col::Column) = num_offdiagonals(col.operator, col.address, col.map)

Base.show(io::IO, col::Column) = print(io, "column(", col.operator, ", ", col.address, ")")

function column(op::AbstractOperator{A}, add::A) where {A}
    map = OccupiedModeMap(add)
    return Column(op, add, map)
end

struct ColOffdiagonals{A,T,C<:AbstractColumn{A,T}} <: AbstractVector{Tuple{A,T}}
    column::C
end
Base.size(od::ColOffdiagonals) = (num_offdiagonals(od.column),)
Base.getindex(od::ColOffdiagonals, i) = get_offdiagonal(od.column, i)

# TODO: name
"""
    CompositeColumn

Column for an operator with a [`CompositeFS`](@ref) address.
"""
struct CompositeColumn{N,A,T,O<:AbstractOperator{A,T},M} <: AbstractColumn{A,T}
    operator::O
    address::A
    maps::M
end
function column(op::AbstractOperator{A,T}, add::A) where {N,A<:CompositeFS{N},T}
    maps = map(OccupiedModeMap, add.components)
    return CompositeColumn{N,A,T,typeof(op),typeof(maps)}(op, add, maps)
end

Base.show(io::IO, col::CompositeColumn) = print(io, "column($(col.operator), $(col.address))")

# NOTE: the following @generated functions take care of calling the operator on all
# combinations of addresses in the `CompositeFS`. This way, it's enough for operators to
# define the 6-argument get_offdiagonal.
function get_offdiagonal(col::CompositeColumn, chosen)
    return get_offdiagonal(CompositeAction(col.operator), col, chosen)
end
@generated function get_offdiagonal(::A, col::CompositeColumn{N}, chosen) where {A,N}
    expr = quote
        orig_chosen = chosen # for error at the end
        add = col.address
        adds = col.address.components
        maps = col.maps
        op = col.operator
    end
    # Per-component terms.
    for i in 1:N
        expr = quote
            $expr
            num = num_offdiagonals(op, adds[$i], maps[$i])
            if chosen ≤ num
                new_add_i, value = get_offdiagonal(op, adds[$i], maps[$i], chosen, $i)
                new_add = update_component(add, new_add_i, Val($i))
                return new_add, value
            else
                chosen -= num
            end
        end
    end
    # Cross-component terms.
    if A ≢ NoCompositeAction
        for i in 1:N, j in 1:N
            i == j && continue
            j > i && A === OneWayCompositeAction && continue
            expr = quote
                $expr
                num = num_offdiagonals(op, adds[$i], adds[$j], maps[$i], maps[$j])
                if chosen ≤ num
                    new_add_i, new_add_j, value = get_offdiagonal(
                        op, adds[$i], adds[$j], maps[$i], maps[$j], chosen, ($i, $j)
                    )
                    new_add = update_component(add, new_add_i, new_add_j, Val($i), Val($j))
                    return new_add, value
                else
                    chosen -= num
                end
            end
        end
    end

    return quote
        $expr
        throw(BoundsError(col, orig_chosen))
    end
end

function num_offdiagonals(col::CompositeColumn)
    return num_offdiagonals(CompositeAction(col.operator), col)
end
@generated function num_offdiagonals(::A, col::CompositeColumn{N}) where {A,N}
    expr = quote
        adds = col.address.components
        maps = col.maps
        op = col.operator
        result = 0
    end
    for i in 1:N
        expr = quote
            $expr
            result += num_offdiagonals(op, adds[$i], maps[$i])
        end
    end
    if A ≢ NoCompositeAction
        for i in 1:N, j in 1:N
            i == j && continue
            j > i && A === OneWayCompositeAction && continue
            expr = quote
                $expr
                result += num_offdiagonals(op, adds[$i], adds[$j], maps[$i], maps[$j])
            end
        end
    end
    return quote
        $expr
        return result
    end
end

function diagonal_element(col::CompositeColumn)
    return diagonal_element(CompositeAction(col.operator), col)
end
@generated function diagonal_element(::A, col::CompositeColumn{N,<:Any,T}) where {A,N,T}
    expr = quote
        adds = col.address.components
        maps = col.maps
        op = col.operator
        result = zero($T)
    end
    for i in 1:N
        expr = quote
            $expr
            result += diagonal_element(op, adds[$i], maps[$i], $i)
        end
    end
    if A ≢ NoCompositeAction
        for i in 1:N, j in 1:N
            i == j && continue
            j > i && A === OneWayCompositeAction && continue
            expr = quote
                $expr
                result += diagonal_element(
                    op, adds[$i], adds[$j], maps[$i], maps[$j], ($i,$j)
                )
            end
        end
    end
    return quote
        $expr
        return result
    end
end
