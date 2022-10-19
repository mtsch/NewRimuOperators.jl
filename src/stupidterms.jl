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




struct StupidTwoBodyTerm{T,F,Adjoint} <: AbstractTerm{T,2}
    fun::F
end
function StupidTwoBodyTerm(fun, adjoint=false)
    T = float(typeof(fun(1, 1, 1, 1, 1, 1)))
    return StupidTwoBodyTerm{T,typeof(fun),adjoint}(fun)
end

LOStructure(::StupidTwoBodyTerm) = AdjointKnown()
isadjoint(::StupidTwoBodyTerm{<:Any,<:Any,A}) where {A} = A
function Base.adjoint(op::StupidTwoBodyTerm{T,F,A}) where {T,F,A}
    return StupidTwoBodyTerm{T,F,!A}(op.fun)
end

# Single fermionic component has no contributions
num_offdiagonals(::StupidTwoBodyTerm, ::FermiFS, _) = 0
diagonal_element(::StupidTwoBodyTerm{T}, ::FermiFS, map, c=1) where {T} = zero(T)

function num_offdiagonals(::StupidTwoBodyTerm, add::BoseFS, map)
    N = length(map)
    M = num_modes(add)

    return N * N * M * M
end

function get_offdiagonal(op::StupidTwoBodyTerm, add::BoseFS, map, chosen, σ=1)
    N = length(map)
    M = num_modes(add)
    p_i, q_i, r_i, s_i = Tuple(CartesianIndices((N,N,M,M))[chosen])

    p = map[p_i]
    q = map[q_i]
    r = find_mode(add, r_i)
    s = find_mode(add, s_i)

    new_add, val = excitation(add, (s,r), (q,p))
    fun_val = op.fun(σ, σ, p.mode, q.mode, r.mode, s.mode)
    if new_add == add
        return add, zero(eltype(op))
    else
        return new_add, val * fun_val / 2
    end
end

function num_offdiagonals(::StupidTwoBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * M * M
end

function get_offdiagonal(op::StupidTwoBodyTerm, add_a, add_b, map_a, map_b, i, (σ, τ))
    N1 = length(map_a)
    N2 = length(map_b)
    M = num_modes(add_a)
    s_i, r_i, q_i, p_i = Tuple(CartesianIndices((M, M, N2, N1))[i])
    p = map_a[p_i]
    q = map_b[q_i]
    r = find_mode(add_b, r_i)
    s = find_mode(add_a, s_i)

    new_add_a, val_a = excitation(add_a, (s,), (p,))
    new_add_b, val_b = excitation(add_b, (r,), (q,))

    if new_add_a == add_a && new_add_b == add_b
        return add_a, add_b, zero(eltype(op))
    else
        val = val_a * val_b * +(
            op.fun(σ, τ, p.mode, q.mode, r.mode, s.mode),
            op.fun(τ, σ, q.mode, p.mode, s.mode, r.mode),
        ) / 2
        return new_add_a, new_add_b, val
    end
end

function diagonal_element(op::StupidTwoBodyTerm, add::BoseFS, map, σ=1)
    return two_body_diagonal(op, map, σ)
end

function diagonal_element(op::StupidTwoBodyTerm, _, _, map_a, map_b, comps)
    return two_body_diagonal(op, map_a, map_b, comps)
end


function Rimu.BasisSetRep(term::AbstractTerm, add; kwargs...)
    return BasisSetRep(SingleTermOperator(term, add); kwargs...)
end
