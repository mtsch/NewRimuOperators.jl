using Rimu.Hamiltonians: transcorrelated_three_body_excitation

binomial3(k) = k * (k - 1) * (k - 2) ÷ 6
#permutations((a,b)::NTuple{2}) = ((a,b), (b,a))
#permutations((a,b,c)::NTuple{3}) = ((a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a))

#conj_3body(fun, σ, τ, ν, p, q, r, s, t, u) = fun(σ, τ, ν, u, t, s, r, q, p)

"""
    MomentumThreeBodyTerm(fun)

The momentum-preserving three body term:

```math
\\sum_{σ,τ,p,q,r,k,k'} f(σ,τ,p,q,q+k',p-k)
    \\hat{a}^†_{p-k,σ} \\hat{a}^†_{q+k',σ} \\hat{a}^†_{r+k-k',τ}
    \\hat{a}_{r,τ} \\hat{a}_{q,σ} \\hat{a}_{p,σ},
```
"""
struct MomentumThreeBodyTerm{T,F,Adjoint} <: AbstractTerm{T,3}
    fun::F
end

function MomentumThreeBodyTerm(fun, adjoint=false)
    T = float(typeof(fun(1, 1, 1, 1, 1, 1, 1, 1, 1)))
    return MomentumThreeBodyTerm{T,typeof(fun),adjoint}(fun)
end

LOStructure(::MomentumThreeBodyTerm) = AdjointKnown()
isadjoint(::MomentumThreeBodyTerm{<:Any,<:Any,A}) where {A} = A
function Base.adjoint(op::MomentumThreeBodyTerm{T,F,A}) where {T,F,A}
    return MomentumThreeBodyTerm{T,F,!A}(op.fun)
end

# Single fermionic component has no contributions
num_offdiagonals(::MomentumThreeBodyTerm, ::FermiFS, _) = 0
diagonal_element(::MomentumThreeBodyTerm{T}, ::FermiFS, map, c=1) where {T} = zero(T)

###
### Bosons -> Self
###
function num_offdiagonals(::MomentumThreeBodyTerm, add::BoseFS, map)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    triplies = count(i -> i.occnum ≥ 3, map)
    M = num_modes(add)

    return (binomial3(singlies) + doublies * (singlies - 1) + triplies) *
        (M * (M + 1)) ÷ 2
end
function _pick_stu(add, p, q, r, dst_index)
    u_mode, t_mode = index_to_sorted_pair(dst_index)
    t_mode -= 1
    k1 = u_mode - p.mode
    k2 = t_mode - q.mode
    s_mode = r.mode - k1 - k2
    if 0 < s_mode ≤ num_modes(add)
        return find_mode(add, (s_mode, t_mode, u_mode))
    else
        return nothing
    end
end
function get_offdiagonal(op::MomentumThreeBodyTerm, add::BoseFS, map, chosen, σ=1)
    M = num_modes(add)
    num_destinations = (M * (M + 1)) ÷ 2
    singlies = length(map)
    double = chosen - binomial3(singlies) * num_destinations

    # First, we determine p, q, r and the dst_index
    if double > 0
        # Will pick doubly + singly or triply occupied modes.
        doublies = count(i -> i.occnum ≥ 2, map)
        triple = double - doublies * (singlies - 1) * num_destinations
        if triple > 0
            src_index, dst_index = fldmod1(triple, num_destinations)
            p, _ = pick_multiply_occupied_mode(add, map, src_index, 3)
            stu = _pick_stu(add, p, p, p, dst_index)
            if isnothing(stu)
                return add, zero(eltype(op))
            else
                s, t, u = stu
            end
            q = r = p

            if !isadjoint(op)
                fun_val = +(
                    op.fun(σ, σ, σ, p.mode, p.mode, p.mode, s.mode, t.mode, u.mode),
                    op.fun(σ, σ, σ, p.mode, p.mode, p.mode, s.mode, u.mode, t.mode),
                )
            else
                fun_val = conj(+(
                    op.fun(σ, σ, σ, u.mode, t.mode, s.mode, p.mode, p.mode, p.mode),
                    op.fun(σ, σ, σ, t.mode, u.mode, s.mode, p.mode, p.mode, p.mode),
                ))
            end
            if t.mode == u.mode
                fun_val /= 2
            end
        else
            src_double, src_single, dst_index = Tuple(CartesianIndices(
                (doublies, singlies - 1, num_destinations)
            )[double])
            p, p_index = pick_multiply_occupied_mode(add, map, src_double, 2)
            src_single += src_single ≥ p_index
            q = map[src_single]
            stu = _pick_stu(add, p, p, q, dst_index)
            if isnothing(stu)
                return add, zero(eltype(op))
            else
                s, t, u = stu
            end
            r = p
            @show p.mode,q.mode,s.mode,t.mode,u.mode

            if !isadjoint(op)
                fun_val = +(
                    op.fun(σ, σ, σ, p.mode, p.mode, q.mode, s.mode, t.mode, u.mode),
                    op.fun(σ, σ, σ, p.mode, q.mode, p.mode, s.mode, t.mode, u.mode),
                    op.fun(σ, σ, σ, q.mode, p.mode, p.mode, s.mode, t.mode, u.mode),
                    op.fun(σ, σ, σ, p.mode, p.mode, q.mode, s.mode, u.mode, t.mode),
                    op.fun(σ, σ, σ, p.mode, q.mode, p.mode, s.mode, u.mode, t.mode),
                    op.fun(σ, σ, σ, q.mode, p.mode, p.mode, s.mode, u.mode, t.mode),
                )
            else
                fun_val = conj(+(
                    op.fun(σ, σ, σ, u.mode, t.mode, s.mode, q.mode, q.mode, p.mode),
                    op.fun(σ, σ, σ, u.mode, t.mode, s.mode, p.mode, q.mode, p.mode),
                    op.fun(σ, σ, σ, u.mode, t.mode, s.mode, p.mode, p.mode, q.mode),
                    op.fun(σ, σ, σ, t.mode, u.mode, s.mode, q.mode, q.mode, p.mode),
                    op.fun(σ, σ, σ, t.mode, u.mode, s.mode, p.mode, q.mode, p.mode),
                    op.fun(σ, σ, σ, t.mode, u.mode, s.mode, p.mode, p.mode, q.mode),
                ))
            end
            if t.mode == s.mode
                fun_val *= 2
            end
        end
    else
        src_index, dst_index = fldmod1(chosen, num_destinations)
        p_index, q_index, r_index = index_to_sorted_triple(src_index, singlies)

        p, q, r = (map[p_index], map[q_index], map[r_index])
        stu = _pick_stu(add, p, q, r, dst_index)
        if isnothing(stu)
            return add, zero(eltype(op))
        else
            s, t, u = stu
        end

        if !isadjoint(op)
            fun_val = +(
                op.fun(σ, σ, σ, p.mode, q.mode, r.mode, s.mode, t.mode, u.mode),
                op.fun(σ, σ, σ, q.mode, p.mode, r.mode, s.mode, t.mode, u.mode),
                op.fun(σ, σ, σ, r.mode, p.mode, q.mode, s.mode, t.mode, u.mode),
                op.fun(σ, σ, σ, p.mode, r.mode, q.mode, s.mode, t.mode, u.mode),
                op.fun(σ, σ, σ, q.mode, r.mode, p.mode, s.mode, t.mode, u.mode),
                op.fun(σ, σ, σ, r.mode, q.mode, p.mode, s.mode, t.mode, u.mode),

                op.fun(σ, σ, σ, p.mode, q.mode, r.mode, s.mode, u.mode, t.mode),
                op.fun(σ, σ, σ, q.mode, p.mode, r.mode, s.mode, u.mode, t.mode),
                op.fun(σ, σ, σ, r.mode, p.mode, q.mode, s.mode, u.mode, t.mode),
                op.fun(σ, σ, σ, p.mode, r.mode, q.mode, s.mode, u.mode, t.mode),
                op.fun(σ, σ, σ, q.mode, r.mode, p.mode, s.mode, u.mode, t.mode),
                op.fun(σ, σ, σ, r.mode, q.mode, p.mode, s.mode, u.mode, t.mode),
            )
        else
            fun_val = conj(+(
                op.fun(σ, σ, σ, u.mode, t.mode, s.mode, r.mode, q.mode, p.mode),
                op.fun(σ, σ, σ, u.mode, t.mode, s.mode, r.mode, p.mode, q.mode),
                op.fun(σ, σ, σ, u.mode, t.mode, s.mode, q.mode, p.mode, r.mode),
                op.fun(σ, σ, σ, u.mode, t.mode, s.mode, q.mode, r.mode, p.mode),
                op.fun(σ, σ, σ, u.mode, t.mode, s.mode, p.mode, r.mode, q.mode),
                op.fun(σ, σ, σ, u.mode, t.mode, s.mode, p.mode, q.mode, r.mode),
            ))
        end
    end

    new_add, val = excitation(add, (u, t, s), (r, q, p))
    if new_add == add
        return add, zero(eltype(op))
    else
        return new_add, val * fun_val
    end
end
function diagonal_element(op::MomentumThreeBodyTerm, add::BoseFS, map, σ=1)
    res = zero(eltype(op))
    for p in map, q in map, r in map
        val = p.occnum * (q.occnum - (q == p)) * (r.occnum - (r == p) - (r == q))
        res += val * +(
            op.fun(σ,σ,σ, p.mode,q.mode,r.mode, r.mode,q.mode,p.mode),
            op.fun(σ,σ,σ, p.mode,q.mode,r.mode, r.mode,p.mode,q.mode),
            op.fun(σ,σ,σ, p.mode,q.mode,r.mode, p.mode,p.mode,r.mode),
            op.fun(σ,σ,σ, p.mode,q.mode,r.mode, q.mode,r.mode,p.mode),
        ) * p.occnum * q.occnum * r.occnum
    end
    return res
end

###
### Bosons -> Any
###
function num_offdiagonals(op::MomentumThreeBodyTerm, add_a::BoseFS, _, map_a, map_b)
    M = num_modes(add_a)
    singlies = length(map_a)
    doublies = count(i -> i.occnum ≥ 2, map_a)
    targets = length(map_b)
    return ((singlies * (singlies - 1)) ÷ 2 + doublies) * targets * (M * (M + 1)) ÷ 2
end

function get_offdiagonal(
    op::MomentumThreeBodyTerm, add_a::BoseFS, add_b, map_a, map_b, chosen, (σ, τ)
)
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)
    double = chosen - N1 * (N1 - 1) * N2 * M * (M + 1) ÷ 4
    if double > 0
        # a†_u a†_t a†_s a_r a_p a_p
        src_index_a, rest = fldmod1(double, N2 * M * (M + 1) ÷ 2)
        src_index_b, dst_index = fldmod1(rest, M * (M + 1) ÷ 2)
        p, _ = pick_multiply_occupied_mode(add_a, map_a, src_index_a, 2)

        t_mode, u_mode = index_to_sorted_pair(dst_index)
        u_mode -= 1
        r = map_b[src_index_b]
        t, u = find_mode(add_a, (t_mode, u_mode))

        k1 = u.mode - p.mode
        k2 = t.mode - p.mode
        s_mode = r.mode - k1 - k2
        if (k1 ≠ 0 || k2 ≠ 0) && 0 < s_mode ≤ M
            s = find_mode(add_b, s_mode)
            new_add_a, val_a = excitation(add_a, (u,t), (p,p))
            new_add_b, val_b = excitation(add_b, (s,), (r,))

            if !isadjoint(op)
                fun_val = +(
                    op.fun(σ, σ, τ, p.mode, p.mode, r.mode, s.mode, t.mode, u.mode),
                    op.fun(σ, σ, τ, p.mode, p.mode, r.mode, s.mode, u.mode, t.mode),
                )
            else
                fun_val = conj(+(
                    op.fun(σ, σ, τ, u.mode, t.mode, s.mode, r.mode, p.mode, p.mode),
                    op.fun(σ, σ, τ, t.mode, u.mode, s.mode, r.mode, p.mode, p.mode),
                ))
            end
            return new_add_a, new_add_b, val_a * val_b * fun_val
        else
            return add_a, add_b, zero(eltype(op))
        end
    else
        # a†_u a†_t a†_s a_r a_q a_p
        src_index_a, src_index_b, dst_index = Tuple(CartesianIndices(
            (N1 * (N1 - 1) ÷ 2, N2, M * (M + 1) ÷ 2)
        )[chosen])
        src_1, src_2 = index_to_sorted_pair(src_index_a)
        p = map_a[src_1]
        q = map_a[src_2]
        r = map_b[src_index_b]

        t_mode, u_mode = index_to_sorted_pair(dst_index)
        u_mode -= 1
        t, u = find_mode(add_a, (t_mode, u_mode))

        k1 = u.mode - p.mode
        k2 = t.mode - q.mode
        s_mode = r.mode - k1 - k2
        if 0 < s_mode ≤ M
            s = find_mode(add_b, s_mode)
            new_add_a, val_a = excitation(add_a, (u,t), (q,p))
            new_add_b, val_b = excitation(add_b, (s,), (r,))
            if new_add_a == add_a && new_add_b == add_b
                return add_a, add_b, zero(eltype(op))
            else
                if !isadjoint(op)
                    fun_val = +(
                        op.fun(σ, σ, τ, p.mode, q.mode, r.mode, s.mode, t.mode, u.mode),
                        op.fun(σ, σ, τ, p.mode, q.mode, r.mode, s.mode, u.mode, t.mode),
                        op.fun(σ, σ, τ, q.mode, p.mode, r.mode, s.mode, t.mode, u.mode),
                        op.fun(σ, σ, τ, q.mode, p.mode, r.mode, s.mode, u.mode, t.mode),
                    ) * 3
                else
                    fun_val = conj(+(
                        op.fun(σ, σ, τ, u.mode, t.mode, s.mode, r.mode, q.mode, p.mode),
                        op.fun(σ, σ, τ, t.mode, u.mode, s.mode, r.mode, q.mode, p.mode),
                        op.fun(σ, σ, τ, u.mode, t.mode, s.mode, r.mode, p.mode, q.mode),
                        op.fun(σ, σ, τ, t.mode, u.mode, s.mode, r.mode, p.mode, q.mode),
                    )) * 3
                end
                return new_add_a, new_add_b, val_a * val_b * fun_val
            end
        else
            return add_a, add_b, zero(eltype(op))
        end
    end
end

###
### Fermions -> Any
###
function num_offdiagonals(::MomentumThreeBodyTerm, add_a::FermiFS, add_b, map_a, map_b)
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)

    return ((N1 - 1) * N1) ÷ 2 * N2 * (M * (M - 1)) ÷ 2
end

function get_offdiagonal(
    op::MomentumThreeBodyTerm, add_a::FermiFS, add_b, map_a, map_b, i, (σ,τ)
)
    M = num_modes(add_a)
    N1 = length(map_a)
    N2 = length(map_b)

    src_a_index, src_b_index, dst_index = Tuple(CartesianIndices(
        (((N1 - 1) * N1) ÷ 2, N2, (M * (M - 1)) ÷ 2)
    )[i])

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
            fun_val = +(
                op.fun(σ, σ, τ, p.mode, q.mode, r.mode, s.mode, t.mode, u.mode),
                op.fun(σ, σ, τ, q.mode, p.mode, r.mode, s.mode, u.mode, t.mode),
                -op.fun(σ, σ, τ, p.mode, q.mode, r.mode, s.mode, u.mode, t.mode),
                -op.fun(σ, σ, τ, q.mode, p.mode, r.mode, s.mode, t.mode, u.mode),
            )
        else
            fun_val = conj(+(
                op.fun(σ, σ, τ, u.mode, t.mode, s.mode, r.mode, q.mode, p.mode),
                op.fun(σ, σ, τ, t.mode, u.mode, s.mode, r.mode, p.mode, q.mode),
                -op.fun(σ, σ, τ, t.mode, u.mode, s.mode, r.mode, q.mode, p.mode),
                -op.fun(σ, σ, τ, u.mode, t.mode, s.mode, r.mode, p.mode, q.mode),
            ))
        end
        return new_add_a, new_add_b, val_a * val_b * fun_val
    else
        return add_a, add_b, zero(eltype(op))
    end
end
function diagonal_element(op::MomentumThreeBodyTerm, add_a, _, map_a, map_b, (σ,τ))
    value = zero(eltype(op))
    for i in 1:length(map_a)
        p = map_a[i]
        if add_a isa BoseFS && p.occnum ≥ 2
            value += sum(map_b) do r
                r.occnum * op.fun(σ, σ, τ, p.mode, p.mode, r.mode, r.mode, p.mode, p.mode)
            end * p.occnum * (p.occnum - 1)
        end
        for j in 1:i-1
            q = map_a[j]
            value += sum(map_b) do r
                val = op.fun(σ, σ, τ, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode) +
                    op.fun(σ, σ, τ, q.mode, p.mode, r.mode, r.mode, p.mode, q.mode)
                val_swap = op.fun(σ, σ, τ, p.mode, q.mode, r.mode, r.mode, p.mode, q.mode) +
                    op.fun(σ, σ, τ, q.mode, p.mode, r.mode, r.mode, q.mode, p.mode)
                if add_a isa FermiFS
                    (val - val_swap) * r.occnum
                else
                    (val + val_swap) * r.occnum
                end
            end * p.occnum * q.occnum
        end
    end
    if !isadjoint(op)
        return value
    else
        return conj(value)
    end
end

###
### Three-component
###
function num_offdiagonals(::MomentumThreeBodyTerm, add_a, add_b, add_c, map_a, map_b, map_c)
    N1 = length(map_a)
    N2 = length(map_b)
    N3 = length(map_c)
    M = num_modes(add_a)
    return N1 * N2 * N3 * M * M
end
function get_offdiagonal(
    op::MomentumThreeBodyTerm, add_a, add_b, add_c, map_a, map_b, map_c, i, (σ,τ,ν)
)
    N1 = length(map_a)
    N2 = length(map_b)
    N3 = length(map_c)
    M = num_modes(add_a)

    p_index, q_index, r_index, t_mode, u_mode = Tuple(CartesianIndices((N1, N2, N3, M, M))[i])

    # Moving from (p,q,r) to s,t
    p = map_a[p_index]
    q = map_b[q_index]
    r = map_c[r_index]
    t = find_mode(add_b, t_mode)
    u = find_mode(add_a, u_mode)

    k1 = u.mode - p.mode
    k2 = t.mode - q.mode
    s_mode = r.mode - k1 - k2
    if 0 < s_mode ≤ M
        s = find_mode(add_c, s_mode)

        new_add_a, val_a = excitation(add_a, (u,), (p,))
        new_add_b, val_b = excitation(add_b, (t,), (q,))
        new_add_c, val_c = excitation(add_c, (s,), (r,))

        if !isadjoint(op)
            fun_val = +(
                op.fun(σ, τ, ν, p.mode, q.mode, r.mode, s.mode, t.mode, u.mode),
                op.fun(σ, ν, τ, p.mode, r.mode, q.mode, t.mode, s.mode, u.mode),
                op.fun(τ, σ, ν, q.mode, p.mode, r.mode, s.mode, u.mode, t.mode),
                op.fun(τ, ν, σ, q.mode, r.mode, p.mode, u.mode, s.mode, t.mode),
                op.fun(ν, σ, τ, r.mode, p.mode, q.mode, u.mode, u.mode, s.mode),
                op.fun(ν, τ, σ, r.mode, q.mode, p.mode, u.mode, t.mode, s.mode),
            )
        else
            fun_val = conj(op.fun(σ, τ, ν, u.mode, t.mode, s.mode, r.mode, q.mode, p.mode))
        end
        if new_add_a == add_a && new_add_b == add_b && new_add_c == add_c
            return add_a, add_b, add_b, zero(eltype(op))
        else
            return new_add_a, new_add_b, new_add_c, fun_val * val_a * val_b * val_c
        end
    else
        return add_a, add_b, add_c, zero(eltype(op))
    end
end

function diagonal_element(op::MomentumThreeBodyTerm, _, _, _, map_a, map_b, map_c, (σ,τ,ν))
    value = zero(eltype(op))
    for p in map_a, q in map_b, r in map_c
        value += +(
            op.fun(σ, τ, ν, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode),
            op.fun(σ, τ, ν, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode),
            op.fun(σ, τ, ν, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode),
            op.fun(σ, τ, ν, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode),
            op.fun(σ, τ, ν, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode),
            op.fun(σ, τ, ν, p.mode, q.mode, r.mode, r.mode, q.mode, p.mode),
        ) * p.occnum * q.occnum * r.occnum
    end
    return value
end

############################################################################################
struct StupidThreeBodyTerm{T,F,Adjoint} <: AbstractTerm{T,3}
    fun::F
end
function StupidThreeBodyTerm(fun, adjoint=false)
    T = float(typeof(fun(1, 1, 1, 1, 1, 1, 1, 1, 1)))
    return StupidThreeBodyTerm{T,typeof(fun),adjoint}(fun)
end

LOStructure(::StupidThreeBodyTerm) = AdjointKnown()
isadjoint(::StupidThreeBodyTerm{<:Any,<:Any,A}) where {A} = A
function Base.adjoint(op::StupidThreeBodyTerm{T,F,A}) where {T,F,A}
    return StupidThreeBodyTerm{T,F,!A}(op.fun)
end

# Single fermionic component has no contributions
num_offdiagonals(::StupidThreeBodyTerm, ::FermiFS, _) = 0
diagonal_element(::StupidThreeBodyTerm{T}, ::FermiFS, map, c=1) where {T} = zero(T)

function num_offdiagonals(::StupidThreeBodyTerm, add::BoseFS, map)
    N = length(map)
    M = num_modes(add)

    return N * N * N * M * M
end
function get_offdiagonal(op::StupidThreeBodyTerm, add::BoseFS, map, chosen, σ=1)
    N = length(map)
    M = num_modes(add)
    # u t s r q p
    p_i, q_i, r_i, t_i, u_i = Tuple(CartesianIndices((N,N,N,M,M))[chosen])

    p = map[p_i]
    q = map[q_i]
    r = map[r_i]
    t = find_mode(add, t_i)
    u = find_mode(add, u_i)

    k1 = u.mode - p.mode
    k2 = t.mode - q.mode
    s = find_mode(add, r.mode - k1 - k2)

    if 0 < s.mode ≤ M
        fun_val = op.fun(σ,σ,σ,p.mode,q.mode,r.mode,s.mode,t.mode,u.mode)
        new_add, val = excitation(add, (u,t,s), (r,q,p))
        if new_add == add
            return add, zero(eltype(op))
        else
            return new_add, val*fun_val
        end
    else
        return add, zero(eltype(op))
    end
end
function diagonal_element(op::StupidThreeBodyTerm, add::BoseFS, map, σ=1)
    res = zero(eltype(op))
    for p in map, q in map, r in map
        val = p.occnum * (q.occnum - (q == p)) * (r.occnum - (r == p) - (r == q))
        res += val * +(
            op.fun(σ,σ,σ, p.mode,q.mode,r.mode, r.mode,q.mode,p.mode),
            op.fun(σ,σ,σ, p.mode,q.mode,r.mode, r.mode,p.mode,q.mode),
        )
    end
    return res
end
