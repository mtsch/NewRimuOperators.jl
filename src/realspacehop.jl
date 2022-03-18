struct RealSpaceHop{A} <: AbstractOperator{A,Float64}
    address::A
    t::Float64
end

starting_address(op::RealSpaceHop) = op.address
function num_offdiagonals(op::RealSpaceHop, _, map)
    return 2 * length(map)
end

function diagonal_element(op::RealSpaceHop, _, _)
    return 0.0
end

function get_offdiagonal(op::RealSpaceHop, add, map, i)
    M = num_modes(add)
    p, k = fldmod1(i, 2)
    src = map[p]
    dst = find_mode(add, mod1(src.mode + ifelse(k == 1, -1, 1), M))
    new_add, value = excitation(add, (dst,), (src,))
    return new_add, -op.t * value
end

CompositeAction(::RealSpaceHop) = NoCompositeAction()
LOStructure(::RealSpaceHop) = IsHermitian()
