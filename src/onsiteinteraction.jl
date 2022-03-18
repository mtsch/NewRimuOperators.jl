struct OnsiteInteraction{A} <: AbstractOperator{A,Float64}
    address::A
    u::Float64
end

starting_address(op::OnsiteInteraction) = op.address
num_offdiagonals(op::OnsiteInteraction, _, _) = 0
num_offdiagonals(op::OnsiteInteraction, _, _, _, _) = 0

function diagonal_element(op::OnsiteInteraction, ::BoseFS, map)
    return op.u * sum(n -> n.occnum * (n.occnum - 1), map) / 2
end
function diagonal_element(op::OnsiteInteraction, ::FermiFS, _)
    return 0.0
end
function diagonal_element(op::OnsiteInteraction, _, _, map_a, map_b)
    N1, N2 = length(map_a), length(map_b)
    i = j = 1
    value = 0.0
    while i ≤ N1 && j ≤ N2
        a = map_a[i]
        b = map_b[j]
        if a.mode == b.mode
            i += 1
            j += 1
            value += a.occnum * b.occnum
        elseif a.mode < b.mode
            i += 1
        else
            j += 1
        end
    end
    return op.u * value
end

CompositeAction(::OnsiteInteraction) = OneWayCompositeAction()
LOStructure(::OnsiteInteraction) = IsHermitian()
