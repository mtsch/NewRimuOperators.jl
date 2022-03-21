"""
    MomentumTransfer(address, f; fold=true, isadjoint=false)

```math
\\sum_{σ,τ,p,q,k} f(σ,τ,p,q,k) a^†_{p + k,σ} a^†{q - k,τ} a_{q,τ} a_{p,σ}
```

In the sum above, ``σ`` and ``τ`` are spin indices, while ``p``, ``q``, and ``k`` are
integer mode indices.
"""
struct MomentumTransfer{A,T,F,Fold,Adjoint} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function MomentumTransfer(
    address::A, fun::F; fold=true, isadjoint=false
) where {A,F,I}
    T = float(typeof(fun(1, 1, 1, 1, 1)))
    return MomentumTransfer{A,T,F,fold,isadjoint}(address, fun)
end
function MomentumTransfer(address, val::Number=1; kwargs...)
    return MomentumTransfer(address, ConstFunction(float(val)); kwargs...)
end

_fold(::MomentumTransfer{<:Any,<:Any,<:Any,F}) where {F} = F
_isadjoint(::MomentumTransfer{<:Any,<:Any,<:Any,<:Any,A}) where {A} = A

function Base.show(io::IO, op::MomentumTransfer)
    print(IOContext(io, :compact => true), "MomentumTransfer(", op.address, ")")
    if op.fun isa ConstFunction
        print(io, "$(op.fun.value);")
    else
        print(io, "$(op.fun);")
    end
    print(io, "fold=$(_fold(op)), isadjoint=$(_isadjoint(op))")
end

starting_address(op::MomentumTransfer) = op.address

# Signle bosonic component
function num_offdiagonals(::MomentumTransfer, bs::BoseFS, map)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    M = num_modes(bs)
    return singlies * (singlies - 1) * (M - 2) + doublies * (M - 1)
end
function get_offdiagonal(op::MomentumTransfer, bs::BoseFS, map, i, comp=1)
    new_add, val, p, q, k = momentum_transfer_excitation(bs, i, map; fold=_fold(op))
    val = if !_isadjoint(op)
        op.fun(comp, comp, p, q, k) * val
    else
        conj(op.fun(comp, comp, p - k, q + k, k)) * val
    end
    return new_add, val/2
end
function diagonal_element(op::MomentumTransfer, bs::BoseFS, map, comp=1)
    onproduct_zero = 0.0
    onproduct_nonzero = 0.0
    for i in 1:length(map)
        occ_i = map[i].occnum
        p = map[i].mode
        onproduct_zero += occ_i * (occ_i - 1) * op.fun(comp, comp, p, p, 0)
        for j in 1:i-1
            occ_j = map[j].occnum
            q = map[j].mode
            k = p - q
            onproduct_nonzero += op.fun(comp, comp, p, q, k) * 4 * occ_i * occ_j
        end
    end
    return (onproduct_zero + onproduct_nonzero) / 2
end

# Single fermionic component has no contributions.
num_offdiagonals(::MomentumTransfer, ::FermiFS, _) = 0
diagonal_element(::MomentumTransfer{<:Any,T}, ::FermiFS, _, _) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::MomentumTransfer, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * (M - 1)
end
function get_offdiagonal(op::MomentumTransfer, add_a, add_b, map_a, map_b, i, (c_a, c_b))
    new_add_a, new_add_b, val, p, q, k = momentum_transfer_excitation(
        add_a, add_b, i, map_a, map_b, fold=_fold(op)
    )
    val = if !_isadjoint(op)
        op.fun(c_a, c_b, p, q, k) * val
    else
        conj(op.fun(c_b, c_a, p - k, q + k, -k)) * val
    end
    return new_add_a, new_add_b, val
end
function diagonal_element(op::MomentumTransfer, _, _, map_a, map_b, (c_a, c_b))
    onproduct_zero = 0.0
    onproduct_nonzero = 0.0
    for i in 1:length(map_a)
        occ_i = map_a[i].occnum
        p = map_a[i].mode
        for j in 1:length(map_b)
            occ_j = map_b[j].occnum
            q = map_b[j].mode
            onproduct_nonzero += (
                (op.fun(c_a, c_b, p, q, 0) + op.fun(c_b, c_a, q, p, 0))/2 * occ_i * occ_j
            )
        end
    end
    return onproduct_nonzero
end

LOStructure(::MomentumTransfer{<:Any,<:Any,<:ConstFunction{<:Real}}) =
    IsHermitian()
LOStructure(::MomentumTransfer{<:Any,<:Any,<:InteractionMatrixFunction{<:Any,<:Real}}) =
    IsHermitian()
LOStructure(::MomentumTransfer) =
    AdjointKnown()

function Base.adjoint(op::MomentumTransfer{A,T,F,Fold,Adjoint}) where {A,T,F,Fold,Adjoint}
    return MomentumTransfer{A,T,F,Fold,!Adjoint}(op.address, op.fun)
end

CompositeAction(::MomentumTransfer) = OneWayCompositeAction()
