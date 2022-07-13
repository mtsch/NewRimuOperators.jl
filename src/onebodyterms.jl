"""
    ParticleCountTerm(fun)

The particle count term:

```math
\\sum_{σ,p} f(σ,p) \\hat{n}_{p,σ},
```
where ``f`` is the `fun`, ``σ`` the spin (component) index, and ``p`` the mode. This term
is diagonal.
"""
struct ParticleCountTerm{F,T,A} <: AbstractTerm{T,1}
    fun::F
end
function ParticleCountTerm(fun::F, adjoint=false) where {F}
    T = float(typeof(fun(1, 1)))
    return ParticleCountTerm{F,T, adjoint}(fun)
end

LOStructure(::ParticleCountTerm{<:Any,<:Real}) = IsDiagonal()
LOStructure(::ParticleCountTerm{<:Any,<:Complex}) = AdjointKnown()
isadjoint(::ParticleCountTerm{<:Any,<:Any,A}) where {A} = A
function LinearAlgebra.adjoint(op::ParticleCountTerm)
    return ParticleCountTerm(op.fun, !isadjoint(op))
end

num_offdiagonals(::ParticleCountTerm, args...) = 0

function diagonal_element(op::ParticleCountTerm, address, map, comp=1)
    result = sum(map) do index
        index.occnum * op.fun(comp, index.mode)
    end
    if isadjoint(op)
        return conj(result)
    else
        return result
    end
end

"""
    NeighbourOneBodyTerm(fun)

The neighbour hopping term in 1D:

```math
\\sum_{σ,⟨j, i⟩} f(σ) \\hat{a}^†_j \\hat{a}_i,
```

where ``f`` is the `fun` and ``sigma`` the spin component index.
"""
struct NeighbourOneBodyTerm{F,T,A} <: AbstractTerm{T,1}
    fun::F
end
function NeighbourOneBodyTerm(fun::F, adjoint=false) where {F}
    T = float(typeof(fun(1)))
    return NeighbourOneBodyTerm{F,T,adjoint}(fun)
end

LOStructure(::NeighbourOneBodyTerm{<:Any,<:Real}) = IsHermitian()
LOStructure(::NeighbourOneBodyTerm{<:Any,<:Complex}) = AdjointKnown()
isadjoint(op::NeighbourOneBodyTerm{<:Any,<:Any,A}) where {A} = A

function LinearAlgebra.adjoint(op::NeighbourOneBodyTerm{<:Any,<:Complex})
    NeighbourOneBodyTerm(op.fun, !isadjoint(op))
end

function num_offdiagonals(op::NeighbourOneBodyTerm, _, map)
    return 2 * length(map)
end

function diagonal_element(op::NeighbourOneBodyTerm, args...)
    return 0.0
end

function get_offdiagonal(op::NeighbourOneBodyTerm, add, map, i, comp=1)
    M = num_modes(add)
    p, k = fldmod1(i, 2)
    src = map[p]
    dst = find_mode(add, mod1(src.mode + ifelse(k == 1, -1, 1), M))
    new_add, value = excitation(add, (dst,), (src,))
    if isadjoint(op)
        return new_add, -conj(op.fun(comp)) * value
    else
        return new_add, -op.fun(comp) * value
    end
end

"""
    FullOneBodyTerm(fun)

The full one-body term:

```math
\\sum_{σ,p,q} f(σ,p,q) \\hat{a}^†_{σ,q} \\hat{a}_{σ,p}
```

where ``f`` is the `fun`, ``σ`` the spin (component) index, and ``p``, ``q`` the source and
target modes.
"""
struct FullOneBodyTerm{F,T,A} <: AbstractTerm{T,1}
    fun::F
end
function FullOneBodyTerm(fun::F, adjoint=false) where {F}
    T = float(typeof(fun(1, 1, 1)))
    return FullOneBodyTerm{F,T,adjoint}(fun)
end

LOStructure(::FullOneBodyTerm{<:Any,<:Real}) = AdjointKnown()
LOStructure(::FullOneBodyTerm{<:Any,<:Complex}) = AdjointKnown()
isadjoint(::FullOneBodyTerm{<:Any,<:Any,A}) where {A} = A

function Base.adjoint(op::FullOneBodyTerm)
    return FullOneBodyTerm(op.fun, !isadjoint(op))
end

num_offdiagonals(::FullOneBodyTerm, add, map) = length(map) * (num_modes(add) - 1)

function diagonal_element(op::FullOneBodyTerm, add, map, comp=1)
    result = sum(map) do index
        op.fun(comp, index.mode, index.mode) * float(index.occnum)
    end
    if isadjoint(op)
        return conj(result)
    else
        return result
    end
end

function get_offdiagonal(op::FullOneBodyTerm, add, map, i, comp=1)
    M = num_modes(add)
    p, q = fldmod1(i, M - 1)
    p_index = map[p] # p-th occupied mode in add
    q += q ≥ p_index.mode # leave out diagonal matrix element
    q_index = find_mode(add, q) # q-th mode in add (not counting p)
    new_add, value = excitation(add, (q_index,), (p_index,)) # a_q^† a_p |add⟩
    if isadjoint(op)
        return new_add, conj(op.fun(comp, q, p_index.mode)) * value
    else
        return new_add, op.fun(comp, p_index.mode, q) * value
    end
end
