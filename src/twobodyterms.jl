"""
    two_body_diagonal(op, map, comp)

Diagonal contribution from two-body term, where the particles are either swapped or stay
still. This is the same for the full two-body term and the momentum preserving two-body
term.
"""
function two_body_diagonal(op, map, comp)
    onproduct_zero = zero(eltype(op))
    onproduct_nonzero = zero(eltype(op))
    for i in 1:length(map)
        occ_i = map[i].occnum
        p = map[i].mode
        onproduct_zero += occ_i * (occ_i - 1) * op.fun(comp, comp, p, p, p, p)
        for j in 1:i-1
            occ_j = map[j].occnum
            q = map[j].mode
            k = p - q
            onproduct_nonzero += (
                op.fun(comp, comp, p, q, p, q) +
                op.fun(comp, comp, q, p, q, p) +
                op.fun(comp, comp, p, q, q, p) +
                op.fun(comp, comp, q, p, p, q)
            ) * occ_i * occ_j
        end
    end
    if isadjoint(op)
        return conj(onproduct_zero + onproduct_nonzero) / 2
    else
        return (onproduct_zero + onproduct_nonzero) / 2
    end
end
function two_body_diagonal(op, map_a, map_b, (σ, τ))
    onproduct = zero(eltype(op))
    for i in map_a
        for j in map_b
            p = i.mode
            q = j.mode
            onproduct += op.fun(σ, τ, p, q, q, p) * i.occnum * j.occnum
        end
    end
    if !isadjoint(op)
        return onproduct
    else
        return conj(onproduct)
    end
end

"""
     OnsiteInteraction(fun)

The onsite interaciton term:

```math
\\sum_{σ,τ,p} f(σ,τ) \\hat{a}^†_{p,σ} \\hat{a}^†_{p,τ} \\hat{a}_{p,τ} \\hat{a}_{p,σ},
```

where ``f`` is the `fun`, ``σ`` and ``τ`` the spin (component) indices, and ``p`` the mode
index.
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
function LinearAlgebra.adjoint(op::OnsiteInteractionTerm{F,T,A}) where {F,T,A}
    return OnsiteInteractionTerm{F,T,!A}(op.fun)
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
function diagonal_element(op::OnsiteInteractionTerm, _, _, map_a, map_b, (σ, τ))
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
        return conj(op.fun(σ, τ)) * value
    else
        return op.fun(σ, τ) * value
    end
end

"""
    MomentumTwoBodyTerm(f; fold=true)

The momentum transfer term:

```math
\\sum_{σ,τ,p,q,k} f(σ,τ,p,q,q+k,p-k) a^†_{p - k,σ} a^†{q + k,τ} a_{q,τ} a_{p,σ}
```

In the sum above, ``σ`` and ``τ`` are spin indices, while ``p``, ``q``, ``q+k``, and ``p-k``
are integer mode indices.

If `fold` is set, transfers that would go out of the Brillouin zone are folded back in.
"""
struct MomentumTwoBodyTerm{F,T,Fold,Adjoint} <: AbstractTerm{T,2}
    fun::F
end
function MomentumTwoBodyTerm(fun::F; fold=true) where {F}
    T = float(typeof(fun(1, 1, 1, 1, 1, 1)))
    return MomentumTwoBodyTerm{F,T,fold,false}(fun)
end
function MomentumTwoBodyTerm(val::Number=1; kwargs...)
    return MomentumTwoBodyTerm(Returns(float(val)); kwargs...)
end

LOStructure(::MomentumTwoBodyTerm) = AdjointKnown()
isfold(::MomentumTwoBodyTerm{<:Any,<:Any,Fold}) where {Fold} = Fold
isadjoint(::MomentumTwoBodyTerm{<:Any,<:Any,<:Any,Adjoint}) where {Adjoint} = Adjoint

function Base.adjoint(op::MomentumTwoBodyTerm{F,T,Fold,Adjoint}) where {F,T,Fold,Adjoint}
    return MomentumTwoBodyTerm{F,T,Fold,!Adjoint}(op.fun)
end

# Signle bosonic component
function num_offdiagonals(::MomentumTwoBodyTerm, add::BoseFS, map)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    M = num_modes(add)
    return singlies * (singlies - 1) ÷ 2 * (M - 2) + doublies * (M ÷ 2)
end

function get_offdiagonal(op::MomentumTwoBodyTerm, add::BoseFS, map, chosen, comp=1)
    M = num_modes(add)
    singlies = length(map)
    double = chosen - (singlies * (singlies - 1) ÷ 2) * (M - 2)
    if double > 0
        src_index, mom_change = fldmod1(double, fld(M, 2))
        p, _ = pick_multiply_occupied_mode(add, map, src_index, 2)
        r_mode = p.mode - mom_change
        s_mode = p.mode + mom_change

        # Folding
        r_fold = r_mode ≤ 0
        s_fold = s_mode > M
        if !isfold(op) && (r_fold || s_fold)
            return add, zero(eltype(op))
        else
            r_mode += M * r_fold
            s_mode -= M * s_fold

            r = find_mode(add, s_mode)
            s = find_mode(add, r_mode)

            new_add, value = excitation(add, (s,r), (p,p))
            if r_mode ≠ s_mode && (!r_fold  || !s_fold)
                value *= 2
            end
            if !isadjoint(op)
                fun_value = +(
                    op.fun(comp, comp, p.mode, p.mode, r.mode, s.mode),
                    op.fun(comp, comp, p.mode, p.mode, s.mode, r.mode),
                )
            else
                fun_value = conj(+(
                    op.fun(comp, comp, r.mode, s.mode, p.mode, p.mode),
                    op.fun(comp, comp, s.mode, r.mode, p.mode, p.mode),
                ))
            end
            return new_add, fun_value * value / 4
        end
    else
        src_index, dst_index = fldmod1(chosen, M - 2)
        fst, snd = index_to_sorted_pair(src_index)
        p, q = (map[fst], map[snd])
        r = pick_mode_avoiding_sources(add, map, dst_index, (p.mode, q.mode))
        mom_change = q.mode - r.mode

        # check if current selection folds. if it does, try again?
        s_mode = p.mode + mom_change
        if !isfold(op) && !(0 < s_mode ≤ M)
            return add, 0.0, 0, 0, 0
        else
            s = find_mode(add, mod1(p.mode + mom_change, M))
        end

        new_add, value = excitation(add, (s,r), (q,p))

        if !isadjoint(op)
            fun_value = +(
                op.fun(comp, comp, p.mode, q.mode, r.mode, s.mode),
                op.fun(comp, comp, q.mode, p.mode, r.mode, s.mode),
                op.fun(comp, comp, p.mode, q.mode, s.mode, r.mode),
                op.fun(comp, comp, q.mode, p.mode, s.mode, r.mode),
            )
        else
            fun_value = conj(+(
                op.fun(comp, comp, s.mode, r.mode, q.mode, p.mode),
                op.fun(comp, comp, s.mode, r.mode, p.mode, q.mode),
                op.fun(comp, comp, r.mode, s.mode, q.mode, p.mode),
                op.fun(comp, comp, r.mode, s.mode, p.mode, q.mode),
            ))
        end
        return new_add, fun_value * value / 4
    end
end

function diagonal_element(op::MomentumTwoBodyTerm, ::BoseFS, map, comp=1)
    return two_body_diagonal(op, map, comp)
end

# Single fermionic component has no contributions.
num_offdiagonals(::MomentumTwoBodyTerm, ::FermiFS, _) = 0
diagonal_element(::MomentumTwoBodyTerm{<:Any,T}, ::FermiFS, map, comp=1) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::MomentumTwoBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * (M - 1)
end
function get_offdiagonal(
    op::MomentumTwoBodyTerm, add_a, add_b, map_a, map_b, chosen, (σ, τ)
)
    M = num_modes(add_a)
    p, remainder = fldmod1(chosen, (M - 1) * length(map_b))
    s_mode, q = fldmod1(remainder, length(map_b))

    p_index = map_a[p]
    p_mode = p_index.mode
    s_mode += s_mode ≥ p_mode # avoid putting the particle in an occupied site

    mom_change = p_mode - s_mode

    q_index = map_b[q]
    q_mode = q_index.mode
    r_mode = q_mode + mom_change

    if isfold(op)
        s_mode = mod1(s_mode, M)
        r_mode = mod1(r_mode, M) # enforce periodic boundary condition
    elseif !(0 < s_mode ≤ M) || !(0 < r_mode ≤ M)
        return add_a, add_b, zero(eltype(op))
    end

    s_index = find_mode(add_a, s_mode)
    r_index = find_mode(add_b, r_mode)

    new_add_a, val_a = excitation(add_a, (s_index,), (p_index,))
    new_add_b, val_b = excitation(add_b, (r_index,), (q_index,))

    if !isadjoint(op)
        fun_val = (
            op.fun(σ, τ, p_mode, q_mode, r_mode, s_mode) +
            op.fun(τ, σ, q_mode, p_mode, s_mode, r_mode)
        ) / 2
    else
        fun_val = conj(
            op.fun(σ, τ, s_mode, r_mode, q_mode, p_mode) +
            op.fun(τ, σ, r_mode, s_mode, p_mode, q_mode)
        ) / 2
    end
    return new_add_a, new_add_b, fun_val * val_a * val_b
end
function diagonal_element(op::MomentumTwoBodyTerm, _, _, map_a, map_b, comps)
    return two_body_diagonal(op, map_a, map_b, comps)
end

"""
    FullTwoBodyTerm(fun)

```math
\\sum_{σ,τ,p,q,r,s} f(σ,τ,p,q,r,s)
    \\hat{a}^†_{s,σ} \\hat{a}^†{r,τ} \\hat{a}_{q,τ} \\hat{a}_{p,σ}
```

where ``f`` is the `fun`, ``σ`` and ``tau`` the spin component indices, and ``p,q,r,s`` the
modes.

If ``σ = τ`` and the component is fermionic, this term produces no offdiagonals.
"""
struct FullTwoBodyTerm{F,T,Adjoint} <: AbstractTerm{T,2}
    fun::F
end
function FullTwoBodyTerm(fun::F) where {F}
    T = float(typeof(fun(1,1,1,1,1,1)))
    return FullTwoBodyTerm{F,T,false}(fun)
end

LOStructure(::FullTwoBodyTerm) = AdjointKnown()
isadjoint(::FullTwoBodyTerm{<:Any,<:Any,A}) where {A} = A
function Base.adjoint(op::FullTwoBodyTerm{T,F,Adjoint}) where {T,F,Adjoint}
    return FullTwoBodyTerm{T,F,!Adjoint}(op.fun)
end

function num_offdiagonals(::FullTwoBodyTerm, add::BoseFS, map)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    M = num_modes(add)
    return M * (M + 1) ÷ 2 * (doublies + (singlies * (singlies - 1)) ÷ 2)
end
function get_offdiagonal(op::FullTwoBodyTerm, add::BoseFS, map, chosen, comp=1)
    M = num_modes(add)
    singlies = length(map)
    double = chosen - (singlies * (singlies - 1)) ÷ 2 * M * (M + 1) ÷ 2
    if double > 0
        src_index, dst_index = fldmod1(double, M * (M + 1) ÷ 2)
        p, _ = pick_multiply_occupied_mode(add, map, src_index, 2)
        q = p
        factor = 0.5
    else
        src_index, dst_index = fldmod1(chosen, M * (M + 1) ÷ 2)
        fst, snd = index_to_sorted_pair(src_index)
        p, q = (map[fst], map[snd])
        factor = 1.0
    end
    # Subtracting 1 from s picks duplicates as well
    r_mode, s_mode = index_to_sorted_pair(dst_index)
    s_mode -= 1
    if r_mode == s_mode
        factor *= 0.5
    end
    r, s = find_mode(add, (r_mode, s_mode))
    new_add, val = excitation(add, (s,r), (q,p))

    if new_add == add
        fun_val = zero(eltype(op))
    elseif !isadjoint(op)
        fun_val = (
            op.fun(comp, comp, p.mode, q.mode, r.mode, s.mode) +
            op.fun(comp, comp, p.mode, q.mode, s.mode, r.mode) +
            op.fun(comp, comp, q.mode, p.mode, r.mode, s.mode) +
            op.fun(comp, comp, q.mode, p.mode, s.mode, r.mode)
        ) * factor
    else
        fun_val = conj(
            op.fun(comp, comp, s.mode, r.mode, q.mode, p.mode) +
            op.fun(comp, comp, r.mode, s.mode, q.mode, p.mode) +
            op.fun(comp, comp, s.mode, r.mode, p.mode, q.mode) +
            op.fun(comp, comp, r.mode, s.mode, p.mode, q.mode)
        ) * factor
    end
    return new_add, fun_val * val / 2
end
function diagonal_element(op::FullTwoBodyTerm, ::BoseFS, map, comp=1)
    return two_body_diagonal(op, map, comp)
end

# Single fermionic component has no contributions.
num_offdiagonals(::FullTwoBodyTerm, ::FermiFS, map) = 0
diagonal_element(::FullTwoBodyTerm{<:Any,T}, ::FermiFS, _, _) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::FullTwoBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * M * M
end
function get_offdiagonal(op::FullTwoBodyTerm, add_a, add_b, map_a, map_b, i, (σ, τ))
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

    if new_add_a == add_a && new_add_b == add_b
        fun_val = zero(eltype(op))
    elseif !isadjoint(op)
        fun_val = op.fun(σ, τ, p_index.mode, q_index.mode, r, s)
    else
        fun_val = conj(op.fun(σ, τ, s, r, q_index.mode, p_index.mode))
    end
    return new_add_a, new_add_b, fun_val * val_a * val_b
end
function diagonal_element(op::FullTwoBodyTerm, _, _, map_a, map_b, comps)
    return two_body_diagonal(op, map_a, map_b, comps)
end
