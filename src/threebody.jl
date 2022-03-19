using Rimu.Hamiltonians: transcorrelated_three_body_excitation

struct ThreeBodyMomentumTransfer{A,T,F} <: AbstractOperator{A,T}
    address::A
    fun::F
    fold::Bool
    isadjoint::Bool
end

function ThreeBodyMomentumTransfer(address, fun; fold=false, isadjoint=false)
    if fold
        throw(ArgumentError("folding three body terms not implemented"))
    end
    T = float(typeof(fun(1, 1, 1, 1)))
    return ThreeBodyMomentumTransfer{typeof(address),T,typeof(fun)}(
        address, fun, fold, isadjoint
    )
end

starting_address(op::ThreeBodyMomentumTransfer) = op.address

# Single bosonic component not implemented

# Single fermionic component has no contributions
num_offdiagonals(::ThreeBodyMomentumTransfer, ::FermiFS, _) = 0
diagonal_element(::ThreeBodyMomentumTransfer{<:Any,T}, ::FermiFS, _, _) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::ThreeBodyMomentumTransfer, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)

    return (N1 - 1) * N1 * N2 * M^2
end

function get_offdiagonal(
    op::ThreeBodyMomentumTransfer, add_a, add_b, map_a, map_b, i, (c_a, c_b)
)
    # TODO should be p,q,s,k,l - but rimu only returns k and l (momentum changes)
    new_add_a, new_add_b, val, k, l = transcorrelated_three_body_excitation(
        add_a, add_b, i, map_a, map_b
    )
    val *= op.fun(c_a, c_b, k, l)
    # TODO conj if we need it.
    return new_add_a, new_add_b, val
end

function diagonal_element(op::ThreeBodyMomentumTransfer, _, _, map_a, map_b, (c_a, c_b))
    value = 0.0
    for p in 1:length(map_a)
        for q in 1:p-1
            k = map_a[p].mode - map_a[q].mode
            qkk = op.fun(c_a, c_b, -k, k)
            value += 2 * qkk * length(map_b)
        end
    end
    return value
end

LOStructure(::ThreeBodyMomentumTransfer) = IsHermitian()
CompositeAction(::ThreeBodyMomentumTransfer) = TwoWayCompositeAction()
