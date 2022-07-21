using Rimu.Hamiltonians: transcorrelated_three_body_excitation

binomial3(k) = k * (k - 1) * (k - 2) ÷ 6

# might be stupid
function pick_three(add, map, chosen)
    singlies = length(map)
    double = chosen - singlies * (singlies - 1) * (singlies - 2) ÷ 6
    if double > 0
        # Will pick a doubly or triply occupied mode.
        doublies = count(i -> i.occnum ≥ 2, map)
        triple = double - doublies * (singlies - 1)
        if triple > 0
            # Pick a triply occupied mode.
            idx, _ = pick_multi(add, map, triple, 3)
            return (idx, idx, idx)
        else
            # Pick a doubly occupied mode and a singly occupied one.
            double, single = fldmod1(double, singlies - 1)
            double_idx, double = pick_multi(add, map, double, 2)
            if single ≥ double
                single_idx = map[single + 1]
                return (double_idx, double_idx, single_idx)
            else
                single_idx = map[single]
                return (single_idx, double_idx, double_idx)
            end
        end
    else
        # Pick three distinct modes.
        # binary search third
        lo = 2
        hi = singlies
        while lo < hi - 1
            mid = lo + ((hi - lo) >> 0x1)
            go_down = binomial3(mid) < chosen
            lo = ifelse(go_down, mid, lo)
            hi = ifelse(go_down, hi, mid)
        end
        chosen -= binomial3(lo)
        @fastmath snd = floor(Int, (√(8 * chosen) + 1)/2)
        fst = chosen - (snd * (snd - 1)) ÷ 2
        return (map[fst], map[snd + 1], map[lo + 1])
    end
end

"""
    MomentumThreeBodyTerm(fun)

The momentum-preserving three body term:

```math
\\sum_{σ,τ,p,q,r,k,k'} f(σ,τ,k,k')
    \\hat{a}^†_{p-k,σ} \\hat{a}^†_{q+k',σ} \\hat{a}^†_{r+k-k',τ}
    \\hat{a}_{r,τ} \\hat{a}_{q,σ} \\hat{a}_{p,σ},
```

Where...

TODO: make `f` accept `p`, `q`, and `s`?
"""
struct MomentumThreeBodyTerm{T,F,Adjoint} <: AbstractTerm{T,3}
    fun::F
end

function MomentumThreeBodyTerm(fun, adjoint=false)
    T = float(typeof(fun(1, 1, 1, 1, 1, 1, 1, 1)))
    return MomentumThreeBodyTerm{T,typeof(fun),adjoint}(fun)
end

LOStructure(::MomentumThreeBodyTerm) = AdjointKnown()
isadjoint(::MomentumThreeBodyTerm{<:Any,<:Any,A}) where {A} = A
function Base.adjoint(op::MomentumThreeBodyTerm)
    return MomentumThreeBodyTerm(op.fun, !isadjoint(op))
end

# Single bosonic component not implemented
function num_offdiagonals(::MomentumThreeBodyTerm, add::BoseFS, map)
    N = length(map)
    M = num_modes(add)
    return N^3 * M^2
end
function get_offdiagonal(op::MomentumThreeBodyTerm, add::BoseFS, map, i, comp=1)
    # TODO: make more efficient

    N = length(map)
    M = num_modes(add)
    p, q, r, p_k, q_l = Tuple(CartesianIndices((N, N, N, M, M))[i])
    # Moving from (p,q,r) to s,t
    p_index = map[p]
    q_index = map[q]
    r_index = map[r]
    k = p_k - p_index.mode
    l = q_index.mode - q_l
    r_kl = r_index.mode + k - l
    to = find_mode(add, (p_k, q_l, r_kl))
    from = (r_index, q_index, p_index)

    new_add = add
    val = 0.0
    if 0 < p_k ≤ M && 0 < q_l ≤ M && 0 < r_kl ≤ M
        new_add, val = excitation(add, to, from)
    end
    if new_add == add || iszero(val)
        return add, 0.0, 0,0
    else
        return new_add, val * op.fun(comp, k, l)
    end
end

# Single fermionic component has no contributions
num_offdiagonals(::MomentumThreeBodyTerm, ::FermiFS, _) = 0
diagonal_element(::MomentumThreeBodyTerm{T}, ::FermiFS, map, c=1) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::MomentumThreeBodyTerm, add_a::FermiFS, add_b, map_a, map_b)
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)

    return ((N1 - 1) * N1) ÷ 2 * N2 * (M * (M - 1)) ÷ 2
end

function get_offdiagonal(
    op::MomentumThreeBodyTerm, add_a::FermiFS, add_b, map_a, map_b, i, (c_a, c_b)
)
    # TODO should be p,q,s,k,l - but rimu only returns k and l (momentum changes)
    # no, it should be p,q,r,s,t,u - up to the user to extract the momentum changes.
    # doesn't generate excitations when k or l are zero (it was designed for transcorrelated)
    #new_add_a, new_add_b, val, k, l = transcorrelated_three_body_excitation(
    #    add_a, add_b, i, map_a, map_b
    #)
    #val *= op.fun(c_a, c_b, k, l)
    # TODO conj if we need it.

    # a_r a_q a_p
    # a†_u a†_t a†_s
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)

    src_a_index, src_b_index, dst_index = Tuple(CartesianIndices((
        ((N1 - 1) * N1) ÷ 2, N2, (M * (M - 1)) ÷ 2
    ))[i])

    p_index, q_index = index_to_sorted_pair(src_a_index)
    p = map_a[p_index]
    q = map_a[q_index]
    r = map_b[src_b_index]

    dst_index_1, dst_index_2 = index_to_sorted_pair(dst_index)
    u, t = find_mode(add_a, (dst_index_1, dst_index_2))

    k1 = u.mode - p.mode
    k2 = t.mode - q.mode
    s_mode = r.mode - k1 - k2
    if 1 ≤ s_mode ≤ M
        s = find_mode(add_b, s_mode)

        new_add_a, val_a = excitation(add_a, (u,t), (q,p))
        new_add_b, val_b = excitation(add_b, (s,), (r,))

        if new_add_a == add_a && new_add_b == add_b
            return add_a, add_b, zero(eltype(op))
        elseif !isadjoint(op)
            # Note: minus signs come in because we permuted the indices
            fun_val = (
                op.fun(c_a, c_b, p.mode, q.mode, r.mode, s.mode, t.mode, u.mode) +
                op.fun(c_a, c_b, q.mode, p.mode, r.mode, s.mode, u.mode, t.mode) +
                -op.fun(c_a, c_b, p.mode, q.mode, r.mode, s.mode, u.mode, t.mode) +
                -op.fun(c_a, c_b, q.mode, p.mode, r.mode, s.mode, t.mode, u.mode)
            )
        else
            fun_val = conj(
                op.fun(c_a, c_b, u.mode, t.mode, s.mode, r.mode, q.mode, p.mode) +
                op.fun(c_a, c_b, t.mode, u.mode, s.mode, r.mode, p.mode, q.mode) +
                -op.fun(c_a, c_b, t.mode, u.mode, s.mode, r.mode, q.mode, p.mode) +
                -op.fun(c_a, c_b, u.mode, t.mode, s.mode, r.mode, p.mode, q.mode)
            )
        end
        return new_add_a, new_add_b, val_a * val_b * fun_val
    else
        return add_a, add_b, zero(eltype(op))
    end
end

function diagonal_element(op::MomentumThreeBodyTerm, ::FermiFS, _, map_a, map_b, (c_a, c_b))
    value = zero(eltype(op))
    for i in 1:length(map_a)
        p = map_a[i].mode
        for j in 1:i-1
            q = map_a[j].mode
            value += sum(map_b) do r
                op.fun(c_a, c_b, p, q, r.mode, r.mode, q, p) +
                op.fun(c_a, c_b, q, p, r.mode, r.mode, p, q) +
                -op.fun(c_a, c_b, p, q, r.mode, r.mode, p, q) +
                -op.fun(c_a, c_b, q, p, r.mode, r.mode, q, p)
            end
        end
    end
    if !isadjoint(op)
        return value
    else
        return conj(value)
    end
end

# 3-component - not possible yet
function num_offdiagonals(::MomentumThreeBodyTerm, add_a, add_b, add_c, map_a, map_b, map_c)
    N1 = length(map_a)
    N2 = length(map_b)
    N3 = length(map_c)
    M = num_modes(add_a)
    return N1 * N2 * N3 * M * M
end

# TODO ez pz
function get_offdiagonal(
    op::MomentumThreeBodyTerm, add_a, add_b, add_c, map_a, map_b, map_c, i, (c_a, c_b, c_d)
)
    N1 = length(map_a)
    N2 = length(map_b)
    N3 = length(map_c)
    M = num_modes(add_a)

    p, q, r, p_k, q_l = s, r, q, p = Tuple(CartesianIndices((N1, N2, N3, M, M))[i])

    # Moving from (p,q,r) to s,t
    p_index = map_a[p]
    q_index = map_b[q]
    r_index = map_c[r]
    k = p_k - p_index.mode
    l = q_index.mode - q_l
    r_kl = r_index.mode + k - l
    to = find_mode(add, (p_k, q_l, r_kl))

    new_add = add
    val = 0.0
    if 0 < p_k ≤ M && 0 < q_l ≤ M && 0 < r_kl ≤ M
        new_add_a, val_a = excitation(add_a, (p_index,), (from[1],))
        new_add_b, val_b = excitation(add_b, (q_index,), (from[2],))
        new_add_c, val_c = excitation(add_c, (r_index,), (from[3],))

        val = val_a * val_b * val_c * op.fun(c_a, c_b, c_d, k, l)
        return new_add_a, new_add_b, new_add_c, val
    end
end
