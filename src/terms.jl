
"""
    ParticleCountTerm(address, fun)

The particle count term:

```math
\\sum_{σ,p} f(σ,p) \\hat{n}_{p,σ},
```

where ``f`` is the `fun`, ``σ`` the spin (component) index, and ``p`` the mode.
"""
struct ParticleCountTerm{A,F,T<:Real} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function ParticleCountTerm(address::A, fun::F) where {A,F}
    T = float(typeof(fun(1, 1)))
    return ParticleCountTerm{A,F,T}(address, fun)
end

starting_address(op::ParticleCountTerm) = op.address
CompositeAction(::ParticleCountTerm) = NoCompositeAction()
LOStructure(::ParticleCountTerm) = IsHermitian() # TODO should be IsDiagonal?

num_offdiagonals(::ParticleCountTerm, args...) = 0

function diagonal_element(op::ParticleCountTerm, address, map, comp=1)
    return sum(map) do index
        index.occnum * op.fun(comp, index.mode)
    end
end

"""
     OnsiteInteraction(address, fun)

The onsite interaciton term:

```math
\\sum_{σ,τ,p} f(σ,τ) \\hat{a}^†_{p,σ} \\hat{a}^†_{p,τ} \\hat{a}_{p,τ} \\hat{a}_{p,σ},
```

where ``f`` is the `fun`, ``σ`` and ``τ`` the spin (component) indices, and ``p`` the mode.
"""
struct OnsiteInteractionTerm{A,F,T<:Real} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function OnsiteInteractionTerm(address::A, fun::F) where {A,F}
    T = float(typeof(fun(1, 1)))
    return OnsiteInteractionTerm{A,F,T}(address, fun)
end

starting_address(op::OnsiteInteractionTerm) = op.address
CompositeAction(::OnsiteInteractionTerm) = OneWayCompositeAction()
LOStructure(::OnsiteInteractionTerm) = IsHermitian()

num_offdiagonals(op::OnsiteInteractionTerm, _, _) = 0
num_offdiagonals(op::OnsiteInteractionTerm, _, _, _, _) = 0

function diagonal_element(op::OnsiteInteractionTerm, ::BoseFS, map, comp=1)
    return op.fun(comp, comp) * sum(n -> n.occnum * (n.occnum - 1), map) / 2
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
    return op.fun(c_a, c_b) * value
end

"""
    NeighbourOneBodyTerm(address, fun)

The neighbour hopping term in 1D:

```math
\\sum_{σ,⟨i, j⟩} f(σ) \\hat(a)^†_i \\hat{a}_j,
```

where ``f`` is the `fun` and ``sigma`` the spin component index.
"""
struct NeighbourOneBodyTerm{A,F,T<:Real} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function NeighbourOneBodyTerm(address::A, fun::F) where {A,F}
    T = float(typeof(fun(1)))
    return NeighbourOneBodyTerm{A,F,T}(address, fun)
end

starting_address(op::NeighbourOneBodyTerm) = op.address
CompositeAction(::NeighbourOneBodyTerm) = NoCompositeAction()
LOStructure(::NeighbourOneBodyTerm) = IsHermitian()

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
    return new_add, -op.fun(comp) * value
end

"""
    FullOneBodyTerm(address, fun)

The full one-body term:

```math
\\sum_{σ,p,q} f(σ,p,q) \\hat{a}^†_{σ,q} \\hat{a}_{σ,p}
```

where ``f`` is the `fun`, ``σ`` the spin (component) index, and ``p``, ``q`` the source and
target modes.
"""
struct FullOneBodyTerm{A,F,T,ADJ} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function FullOneBodyTerm(address::A, fun::F) where {A,F}
    T = float(typeof(fun(1, 1, 1)))
    return FullOneBodyTerm{A,F,T,false}(address, fun)
end

starting_address(op::FullOneBodyTerm) = op.address
CompositeAction(::FullOneBodyTerm) = NoCompositeAction()

# TODO: break this out or find a way to signal that the function is Hermitian?
function LOStructure(op::FullOneBodyTerm{<:Any,F}) where {F}
    if F <: ConstFunction{<:Real} || F <: ParameterColumnFunction{<:Any,<:Real}
        return IsHermitian()
    else
        return AdjointKnown()
    end
end

isadjoint(::FullOneBodyTerm{<:Any,<:Any,<:Any,ADJ}) where {ADJ} = ADJ

function Base.adjoint(op::FullOneBodyTerm{A,F,T,ADJ}) where {A,F,T,ADJ}
    FullOneBodyTerm{A,F,T,!ADJ}(op.address, op.fun)
end

num_offdiagonals(::FullOneBodyTerm, add, map) = length(map) * (num_modes(add) - 1)

function diagonal_element(op::FullOneBodyTerm, add, map, comp=1)
    return sum(map) do index
        op.fun(comp, index.mode, index.mode) * float(index.occnum)
    end
end

function get_offdiagonal(op::FullOneBodyTerm, add, map, i, comp=1)
    M = num_modes(add)
    p, q = fldmod1(i, M - 1)
    p_index = map[p] # p-th occupied mode in add
    q += q ≥ p_index.mode # leave out diagonal matrix element
    q_index = find_mode(add, q) # q-th mode in add (not counting p)
    new_add, value = excitation(add, (q_index,), (p_index,)) # a_q^† a_p |add⟩
    if !isadjoint(op)
        return new_add, op.fun(comp, p_index.mode, q) * value
    else
        return new_add, conj(op.fun(comp, q, p_index.mode)) * value
    end
end

"""
    MomentumTwoBodyTerm(address, f; fold=true)

The momentum transfer term:

```math
\\sum_{σ,τ,p,q,k} f(σ,τ,p,q,k) a^†_{p + k,σ} a^†{q - k,τ} a_{q,τ} a_{p,σ}
```

In the sum above, ``σ`` and ``τ`` are spin indices, while ``p``, ``q``, and ``k`` are
integer mode indices.

If `fold` is set, transfers that would go out of the Brillouin zone are folded back in.
"""
struct MomentumTwoBodyTerm{A,T,F,Fold,Adjoint} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function MomentumTwoBodyTerm(address::A, fun::F; fold=true) where {A,F}
    T = float(typeof(fun(1, 1, 1, 1, 1)))
    return MomentumTwoBodyTerm{A,T,F,fold,false}(address, fun)
end
function MomentumTwoBodyTerm(address, val::Number=1; kwargs...)
    return MomentumTwoBodyTerm(address, ConstFunction(float(val)); kwargs...)
end

_fold(::MomentumTwoBodyTerm{<:Any,<:Any,<:Any,F}) where {F} = F
isadjoint(::MomentumTwoBodyTerm{<:Any,<:Any,<:Any,<:Any,A}) where {A} = A

function Base.show(io::IO, op::MomentumTwoBodyTerm)
    print(IOContext(io, :compact => true), "MomentumTwoBodyTerm(", op.address, ")")
    if op.fun isa ConstFunction
        print(io, "$(op.fun.value);")
    else
        print(io, "$(op.fun);")
    end
    print(io, "fold=$(_fold(op)), isadjoint=$(isadjoint(op))")
end

starting_address(op::MomentumTwoBodyTerm) = op.address
CompositeAction(::MomentumTwoBodyTerm) = OneWayCompositeAction()

function LOStructure(::MomentumTwoBodyTerm{F}) where {F}
    if F <: ConstFunction{<:Real} || F <: InteractionMatrixFunction{<:Any,<:Real}
        return IsHermitian()
    else
        return AdjointKnown()
    end
end
function Base.adjoint(op::MomentumTwoBodyTerm{A,T,F,FLD,ADJ}) where {A,T,F,FLD,ADJ}
    return MomentumTwoBodyTerm{A,T,F,FLD,!ADJ}(op.address, op.fun)
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
        conj(op.fun(comp, comp, p - k, q + k, k)) * val
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
    return (onproduct_zero + onproduct_nonzero) / 2
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
    FullTwoBodyTerm(address, fun)

```math
\\sum_{σ,τ,p,q,r,s} f(σ,τ,p,q,r,s)
    \\hat{a}^†_{p,σ} \\hat{a}^†{q,τ} \\hat{a}_{r,τ} \\hat{a}_{s,σ}
```

where ``f`` is the `fun`, ``σ`` and ``tau`` the spin component indices, and ``p,q,r,s`` the
modes.

If ``σ = τ`` and the component is fermionic, this term produces no elements.
"""
struct FullTwoBodyTerm{A,F,T,ADJ} <: AbstractOperator{A,T}
    address::A
    fun::F
end
function FullTwoBodyTerm(address::A, fun::F) where {A,F}
    T = float(typeof(fun(1,1,1,1,1,1)))
    return FullTwoBodyTerm{A,F,T}(address, fun)
end

_fold(::FullTwoBodyTerm{<:Any,<:Any,<:Any,F}) where {F} = F
isadjoint(::FullTwoBodyTerm{<:Any,<:Any,<:Any,A}) where {A} = A

function Base.show(io::IO, op::FullTwoBodyTerm)
    print(IOContext(io, :compact => true), "FullTwoBodyTerm(", op.address, ")")
    if op.fun isa ConstFunction
        print(io, "$(op.fun.value);")
    else
        print(io, "$(op.fun);")
    end
    print(io, "fold=$(_fold(op)), isadjoint=$(isadjoint(op))")
end

starting_address(op::FullTwoBodyTerm) = op.address
CompositeAction(::FullTwoBodyTerm) = OneWayCompositeAction()

function LOStructure(::FullTwoBodyTerm{<:Any,F}) where {F}
    if F <: ConstFunction{<:Real} || F <: InteractionMatrixFunction{<:Any,<:Real}
        return IsHermitian()
    else
        return AdjointKnown()
    end
end
function Base.adjoint(op::FullTwoBodyTerm{A,T,F,ADJ}) where {A,T,F,ADJ}
    return FullTwoBodyTerm{A,T,F,!ADJ}(op.address, op.fun)
end

# TODO: Signle bosonic component not implemented

# Single fermionic component has no contributions. (or does it?)
function num_offdiagonals(::FullTwoBodyTerm, ::FermiFS, map, comp=1)
    length(map) * (M - 1)
end
diagonal_element(::FullTwoBodyTerm{<:Any,T}, ::FermiFS, _, _) where {T} = zero(T)

# Cross-component part
function num_offdiagonals(::FullTwoBodyTerm, add_a, add_b, map_a, map_b)
    M = num_modes(add_a)
    return length(map_a) * length(map_b) * (M - 1) * (M - 1)
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
    for i in 1:length(map_a)
        for j in 1:length(map_b)
            p = i.mode
            q = j.mode
            onproduct += op.fun(c_a, c_b, p, q, q, p) * i.occnum * j.occnum
        end
    end
    return onproduct
end

using Rimu.Hamiltonians: transcorrelated_three_body_excitation

"""
    MomentumThreeBodyTerm(address, fun)

The momentum-preserving three body term:

```math
\\sum_{σ,τ,p,q,s,k,k'} f(σ,τ,k,k')
    \\hat{a}^†_{p-k,σ} \\hat{a}^†_{q+k',σ} \\hat{a}^†_{s+k-k',τ}
    \\hat{a}_{s,τ} \\hat{a}_{q,σ} \\hat{a}_{p,σ},
```

Where...

TODO: make `f` accept `p`, `q`, and `s`?
"""
struct MomentumThreeBodyTerm{A,T,F} <: AbstractOperator{A,T}
    address::A
    fun::F
    fold::Bool
    isadjoint::Bool
end

function MomentumThreeBodyTerm(address, fun; fold=false, isadjoint=false)
    if fold
        throw(ArgumentError("folding three body terms not implemented"))
    end
    T = float(typeof(fun(1, 1, 1, 1)))
    return MomentumThreeBodyTerm{typeof(address),T,typeof(fun)}(
        address, fun, fold, isadjoint
    )
end

starting_address(op::MomentumThreeBodyTerm) = op.address
LOStructure(::MomentumThreeBodyTerm) = IsHermitian()
CompositeAction(::MomentumThreeBodyTerm) = TwoWayCompositeAction()

# Single bosonic component not implemented

# Single fermionic component has no contributions
num_offdiagonals(::MomentumThreeBodyTerm, ::FermiFS, _) = 0
diagonal_element(::MomentumThreeBodyTerm{<:Any,T}, ::FermiFS, map, c=1) where {T} = zero(T)

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
