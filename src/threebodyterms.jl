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

function MomentumThreeBodyTerm(fun; fold=false, isadjoint=false)
    if fold
        throw(ArgumentError("folding three body terms not implemented"))
    end
    T = float(typeof(fun(1, 1, 1, 1)))
    return MomentumThreeBodyTerm{T,typeof(fun),isadjoint}(fun)
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
function num_offdiagonals(::MomentumThreeBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)

    return (N1 - 1) * N1 * N2 * M^2
end

function get_offdiagonal(
    op::MomentumThreeBodyTerm, add_a, add_b, map_a, map_b, i, (c_a, c_b)
)
    # TODO should be p,q,s,k,l - but rimu only returns k and l (momentum changes)
    # doesn't generate excitations when k or l are zero (it was designed for transcorrelated)
    new_add_a, new_add_b, val, k, l = transcorrelated_three_body_excitation(
        add_a, add_b, i, map_a, map_b
    )
    val *= op.fun(c_a, c_b, k, l)
    # TODO conj if we need it.
    return new_add_a, new_add_b, val
end

function diagonal_element(op::MomentumThreeBodyTerm, _, _, map_a, map_b, (c_a, c_b))
    value = zero(eltype(op))
    for p in 1:length(map_a)
        for q in 1:p-1
            k = map_a[p].mode - map_a[q].mode
            qkk = op.fun(c_a, c_b, -k, k)
            value += 2 * qkk * length(map_b)
        end
    end
    return value
end

# 3-component - not possible yet
function num_offdiagonals(::MomentumThreeBodyTerm, add_a, add_b, add_c, map_a, map_b, map_c)
    N1 = length(map_a)
    N2 = length(map_b)
    N3 = length(map_c)
    M = num_modes(add_a)
    return N1 * N2 * N3 * M * M
end

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
