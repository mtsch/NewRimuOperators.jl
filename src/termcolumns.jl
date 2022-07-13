"""
    TermColumn <: AbstractColumn

Column for a [`Term`](@ref) with a [`CompositeFS`](@ref) address.
"""
struct TermColumn{N,K,T,A,O<:AbstractTerm{T,K},M} <: AbstractColumn{A,T}
    term::O
    address::A
    maps::M
end
function column(op::AbstractTerm{T,K}, add::A) where {N,K,T,A<:CompositeFS{N}}
    maps = map(OccupiedModeMap, add.components)
    return TermColumn{N,K,T,A,typeof(op),typeof(maps)}(op, add, maps)
end

Base.show(io::IO, col::TermColumn) = print(io, "column($(col.term), $(col.address))")

# NOTE: the following @generated functions take care of calling the operator on all
# combinations of addresses in the `CompositeFS`. This way, it's enough for operators to
# define the 6-argument get_offdiagonal.
@generated function get_offdiagonal(col::TermColumn{N,K}, chosen) where {N,K}
    expr = quote
        orig_chosen = chosen # for error at the end
        add = col.address
        adds = col.address.components
        maps = col.maps
        op = col.term
    end
    # Per-component
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
    # Cross-component terms - only needed for two-body and three-body terms.
    if K ≥ 2
        for i in 1:N, j in 1:N
            i == j && continue
            # If K = 1, this is a two-body term and we only need to generate the offdiagonal
            # in one direction.
            j > i && K == 2 && continue
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
    # Three-component terms.
    if K ≥ 3
        for i in 1:N, j in (i + 1):N, k in (j + 1):N
            expr = quote
                $expr
                used_adds = (adds[$i], adds[$j], adds[$k])
                used_maps = (maps[$i], maps[$j], maps[$k])
                num = num_offdiagonals(op, used_adds..., used_maps...)
                if chosen ≤ num
                    new_add_i, new_add_j, new_add_k, value = get_offdiagonal(
                        op, used_adds..., used_maps..., chosen, ($i, $j, $k)
                    )
                    new_add = update_component(
                        add, new_add_i, new_add_j, new_add_k, Val($i), Val($j), Val($k)
                    )
                    return new_add, value
                else
                    chosen -= num
                end
            end
        end
    end
    if K ≥ 4
        throw(ArgumentError("4-body terms are not supported"))
    end

    return quote
        $expr
        throw(BoundsError(col, orig_chosen))
    end
end

@generated function num_offdiagonals(col::TermColumn{N,K}) where {N,K}
    expr = quote
        adds = col.address.components
        maps = col.maps
        op = col.term
        result = 0
    end
    for i in 1:N
        expr = quote
            $expr
            result += num_offdiagonals(op, adds[$i], maps[$i])
        end
    end
    if K ≥ 2
        for i in 1:N, j in 1:N
            i == j && continue
            j > i && K == 2 && continue
            expr = quote
                $expr
                result += num_offdiagonals(op, adds[$i], adds[$j], maps[$i], maps[$j])
            end
        end
    end
    if K ≥ 3
        for i in 1:N, j in (i + 1):N, k in (j + 1):N
            expr = quote
                $expr
                used_adds = (adds[$i], adds[$j], adds[$k])
                used_maps = (maps[$i], maps[$j], maps[$k])
                result += num_offdiagonals(op, used_adds..., used_maps...)
            end
        end
    end
    if K ≥ 4
        throw(ArgumentError("4-body terms are not supported"))
    end
    return quote
        $expr
        return result
    end
end

@generated function diagonal_element(col::TermColumn{N,K,T}) where {N,K,T}
    expr = quote
        adds = col.address.components
        maps = col.maps
        op = col.term
        result = zero($T)
    end
    for i in 1:N
        expr = quote
            $expr
            result += diagonal_element(op, adds[$i], maps[$i], $i)
        end
    end
    if K ≥ 2
        for i in 1:N, j in 1:N
            i == j && continue
            j > i && K == 2 && continue
            expr = quote
                $expr
                result += diagonal_element(
                    op, adds[$i], adds[$j], maps[$i], maps[$j], ($i,$j)
                )
            end
        end
    end
    if K ≥ 3
        for i in 1:N, j in (i + 1):N, k in (j + 1):N
            expr = quote
                $expr
                used_adds = (adds[$i], adds[$j], adds[$k])
                used_maps = (maps[$i], maps[$j], maps[$k])
                result += diagonal_element(op, used_adds..., used_maps..., ($i, $j, $k))
            end
        end
    end
    if K ≥ 4
        throw(ArgumentError("4-body terms are not supported"))
    end
    return quote
        $expr
        return result
    end
end
