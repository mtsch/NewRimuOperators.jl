"""
    MomentumTransfer(address, fun; fold=true, isadjoint=false)

```math
\\sum_{p,q,k} f(k) a^†_{p + k} a^†{q - k} a_q a_p
```
"""
struct MomentumTransfer{A,T,F} <: AbstractOperator{A,T}
    address::A
    fun::F
    fold::Bool
    isadjoint::Bool
end
function MomentumTransfer(address, fun; fold=true, isadjoint=false)
    T = float(typeof(fun(1)))
    return MomentumTransfer{typeof(address), T, typeof(fun)}(address, fun, fold, isadjoint)
end
function MomentumTransfer(address, val::Number=1; kwargs...)
    return MomentumTransfer(address, ConstFunction(float(val)); kwargs...)
end
function Base.show(io::IO, op::MomentumTransfer)
    print(io, "MomentumTransfer($(op.address), ")
    if op.fun isa ConstFunction
        print(io, "$(op.fun.value); fold=$(op.fold))")
    else
        print(io, "$(op.fun); fold=$(op.fold), isadjoint=$(op.isadjoint))")
    end
end

starting_address(op::MomentumTransfer) = op.address

# Signle bosonic component
function num_offdiagonals(::MomentumTransfer, bs::BoseFS, map)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    M = num_modes(bs)
    return singlies * (singlies - 1) * (M - 2) + doublies * (M - 1)
end
function get_offdiagonal(op::MomentumTransfer, bs::BoseFS, map, i)
    new_add, val, mom_change = momentum_transfer_excitation(bs, i, map; fold=op.fold)
    val = op.isadjoint ? op.fun(mom_change) * val : conj(op.fun(-mom_change)) * val
    return new_add, val
end
function diagonal_element(op::MomentumTransfer, bs::BoseFS, map)
    onproduct_zero = 0.0
    onproduct_nonzero = 0.0
    for i in 1:length(map)
        occ_i = map[i].occnum
        p = map[i].mode
        onproduct_zero += occ_i * (occ_i - 1)
        for j in 1:i-1
            occ_j = map[j].occnum
            k = p - map[j].mode
            onproduct_nonzero += op.fun(k) * 4 * occ_i * occ_j
        end
    end
    return onproduct_zero * op.fun(0) + onproduct_nonzero
end

# Single fermionic component has no contributions.
num_offdiagonals(::MomentumTransfer, ::FermiFS, _) = 0
diagonal_element(::MomentumTransfer{<:Any,T}, ::FermiFS, _) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::MomentumTransfer, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * (M - 1)
end
function get_offdiagonal(op::MomentumTransfer, add_a, add_b, map_a, map_b, i)
    new_add_a, new_add_b, val, mom_change = momentum_transfer_excitation(add_a, add_b, i, map_a, map_b)
    val = op.isadjoint ? op.fun(mom_change) * val : conj(op.fun(-mom_change)) * val
    return new_add_a, new_add_b, 2 * val
end
function diagonal_element(op::MomentumTransfer, _, _, map_a, map_b)
    onproduct_zero = 0.0
    onproduct_nonzero = 0.0
    for i in 1:length(map_a)
        occ_i = map_a[i].occnum
        p = map_a[i].mode
        for j in 1:length(map_b)
            occ_j = map_b[j].occnum
            k = p - map_b[j].mode
            onproduct_nonzero += op.fun(k) * occ_i * occ_j
        end
    end
    return 2 * (onproduct_zero * op.fun(0) + onproduct_nonzero)
end

LOStructure(::MomentumTransfer{<:Any,<:Any,<:ConstFunction{<:Real}}) = IsHermitian()
LOStructure(::MomentumTransfer) = AdjointKnown()
function Base.adjoint(op::M) where {M<:MomentumTransfer}
    return M(op.address, op.fun, op.fold, !op.isadjoint)
end

CompositeAction(::MomentumTransfer) = OneWayCompositeAction()
