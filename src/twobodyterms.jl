"""
     OnsiteInteraction(fun)

The onsite interaciton term:

```math
\\sum_{σ,τ,p} f(σ,τ) \\hat{a}^†_{p,σ} \\hat{a}^†_{p,τ} \\hat{a}_{p,τ} \\hat{a}_{p,σ},
```

where ``f`` is the `fun`, ``σ`` and ``τ`` the spin (component) indices, and ``p`` the mode.
"""
struct OnsiteInteractionTerm{F,T,A} <: AbstractTerm{T,2}
    fun::F
end
function OnsiteInteractionTerm(fun::F, adjoint=false) where {F}
    T = float(typeof(fun(1, 1)))
    return OnsiteInteractionTerm{F,T,adjoint}(fun)
end

LOStructure(::OnsiteInteractionTerm{<:Any,<:Real}) = IsDiagonal()
LOStructure(::OnsiteInteractionTerm{<:Any,<:Complex}) = AdjointKnown()
isadjoint(::OnsiteInteractionTerm{<:Any,<:Any,A}) where {A} = A
function LinearAlgebra.adjoint(op::OnsiteInteractionTerm)
    return OnsiteInteractionTerm(op.fun, !isadjoint(op))
end

num_offdiagonals(op::OnsiteInteractionTerm, _, _) = 0
num_offdiagonals(op::OnsiteInteractionTerm, _, _, _, _) = 0

function diagonal_element(op::OnsiteInteractionTerm, ::BoseFS, map, comp=1)
    value = sum(n -> n.occnum * (n.occnum - 1), map) / 2
    if isadjoint(op)
        return conj(op.fun(comp, comp)) * value
    else
        return op.fun(comp, comp) * value
    end
end
function diagonal_element(op::OnsiteInteractionTerm, ::FermiFS, _, _)
    return 0.0
end
function diagonal_element(op::OnsiteInteractionTerm, _, _, map_a, map_b, (c_a, c_b))
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
    if isadjoint(op)
        return conj(op.fun(c_a, c_b)) * value
    else
        return op.fun(c_a, c_b) * value
    end
end

"""
    MomentumTwoBodyTerm(f; fold=true)

The momentum transfer term:

```math
\\sum_{σ,τ,p,q,k} f(σ,τ,p,q,k) a^†_{p + k,σ} a^†{q - k,τ} a_{q,τ} a_{p,σ}
```

In the sum above, ``σ`` and ``τ`` are spin indices, while ``p``, ``q``, and ``k`` are
integer mode indices.

If `fold` is set, transfers that would go out of the Brillouin zone are folded back in.
"""
struct MomentumTwoBodyTerm{F,T,Fold,Adjoint} <: AbstractTerm{T,2}
    fun::F
end
function MomentumTwoBodyTerm(fun::F; fold=true) where {F}
    T = float(typeof(fun(1, 1, 1, 1, 1)))
    return MomentumTwoBodyTerm{F,T,fold,false}(fun)
end
function MomentumTwoBodyTerm(val::Number=1; kwargs...)
    return MomentumTwoBodyTerm(ConstFunction(float(val)); kwargs...)
end

_fold(::MomentumTwoBodyTerm{<:Any,<:Any,Fold}) where {Fold} = Fold
isadjoint(::MomentumTwoBodyTerm{<:Any,<:Any,<:Any,Adjoint}) where {Adjoint} = Adjoint

function Base.adjoint(op::MomentumTwoBodyTerm{F,T,Fold,Adjoint}) where {F,T,Fold,Adjoint}
    return MomentumTwoBodyTerm{F,T,Fold,!Adjoint}(op.fun)
end

# Signle bosonic component
function num_offdiagonals(::MomentumTwoBodyTerm, bs::BoseFS, map)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    M = num_modes(bs)
    return singlies * (singlies - 1) * (M - 2) + doublies * (M - 1)
end

function get_offdiagonal(op::MomentumTwoBodyTerm, bs::BoseFS, map, i, comp=1)
    new_add, val, p, q, k = momentum_transfer_excitation(bs, i, map; fold=_fold(op))
    val = if !isadjoint(op)
        op.fun(comp, comp, p, q, k) * val
    else
        error("adjoint is not supported for bosons")
        conj(op.fun(comp, comp, q + k, p - k, -k)) * val
    end

    return new_add, val/2
end
function diagonal_element(op::MomentumTwoBodyTerm, bs::BoseFS, map, comp=1)
    onproduct_zero = zero(eltype(op))
    onproduct_nonzero = zero(eltype(op))
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
    if isadjoint(op)
        return conj(onproduct_zero + onproduct_nonzero) / 2
    else
        return (onproduct_zero + onproduct_nonzero) / 2
    end
end

# Single fermionic component has no contributions.
num_offdiagonals(::MomentumTwoBodyTerm, ::FermiFS, _) = 0
diagonal_element(::MomentumTwoBodyTerm{<:Any,T}, ::FermiFS, map, comp=1) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::MomentumTwoBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * (M - 1)
end
function get_offdiagonal(op::MomentumTwoBodyTerm, add_a, add_b, map_a, map_b, i, (c_a, c_b))
    new_add_a, new_add_b, val, p, q, k = momentum_transfer_excitation(
        add_a, add_b, i, map_a, map_b, fold=_fold(op)
    )
    val = if !isadjoint(op)
        op.fun(c_a, c_b, p, q, k) * val
    else
        conj(op.fun(c_b, c_a, p - k, q + k, -k)) * val
    end
    return new_add_a, new_add_b, val
end
function diagonal_element(op::MomentumTwoBodyTerm, _, _, map_a, map_b, (c_a, c_b))
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

"""
    FullTwoBodyTerm(fun)

```math
\\sum_{σ,τ,p,q,r,s} f(σ,τ,p,q,r,s)
    \\hat{a}^†_{s,σ} \\hat{a}^†{r,τ} \\hat{a}_{q,τ} \\hat{a}_{p,σ}
```

where ``f`` is the `fun`, ``σ`` and ``tau`` the spin component indices, and ``p,q,r,s`` the
modes.

If ``σ = τ`` and the component is fermionic, this term produces no elements.
"""
struct FullTwoBodyTerm{F,T,Adjoint} <: AbstractTerm{T,2}
    fun::F
end
function FullTwoBodyTerm(fun::F) where {F}
    T = float(typeof(fun(1,1,1,1,1,1)))
    return FullTwoBodyTerm{F,T,false}(fun)
end

isadjoint(::FullTwoBodyTerm{<:Any,<:Any,Adjoint}) where {Adjoint} = Adjoint

function Base.adjoint(op::FullTwoBodyTerm{T,F,Adjoint}) where {T,F,Adjoint}
    return FullTwoBodyTerm{T,F,!Adjoint}(op.fun)
end

# TODO: Signle bosonic component not implemented

# Single fermionic component has no contributions.
num_offdiagonals(::FullTwoBodyTerm, ::FermiFS, map, comp=1) = 0
diagonal_element(::FullTwoBodyTerm{<:Any,T}, ::FermiFS, _, _) where {T} = zero(T)

function full_two_body_excitation(add_a, add_b, i, map_a, map_b)
    # TODO missing one-body term
    N1 = length(map_a)
    N2 = length(map_b)
    M = num_modes(add_a)
    s, r, q, p = Tuple(CartesianIndices((M, M, N2, N1))[i])
    s_index = find_mode(add_a, s)
    r_index = find_mode(add_b, r)
    q_index = map_b[q]
    p_index = map_a[p]

    new_add_a, val_a = excitation(add_a, (s_index,), (p_index,))
    new_add_b, val_b = excitation(add_b, (r_index,), (q_index,))

    return new_add_a, new_add_b, val_a * val_b, p_index.mode, q_index.mode, r, s
end

# Cross-component part
function num_offdiagonals(::FullTwoBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * M * M
end
function get_offdiagonal(op::FullTwoBodyTerm, add_a, add_b, map_a, map_b, i, (c_a, c_b))
    new_add_a, new_add_b, val, p, q, r, s = full_two_body_excitation(
        add_a, add_b, i, map_a, map_b,
    )
    val = if !isadjoint(op)
        op.fun(c_a, c_b, p, q, r, s) * val
    else
        error("TODO")
        #conj(op.fun(c_b, c_a, r, s, p, q)) * val ?
    end
    return new_add_a, new_add_b, val
end
function diagonal_element(
    op::FullTwoBodyTerm, _, _, map_a, map_b, (c_a, c_b)
)
    # This is equivalent to ∑_{p,q} fun(σ,τ,p,q,q,p) n_{p,σ} n_{q,τ}
    onproduct = zero(eltype(op))
    for i in map_a
        for j in map_b
            p = i.mode
            q = j.mode
            onproduct += op.fun(c_a, c_b, p, q, q, p) * i.occnum * j.occnum
        end
    end
    return onproduct
end

using Rimu.Hamiltonians: transcorrelated_three_body_excitation
