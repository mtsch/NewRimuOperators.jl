"""
    index_to_sorted_pair(index)

Map combinatorial `index` to (sorted) pair of (distinct) integers.

# Example

```julia
julia> index_to_sorted_pair(1)
(1, 2)
julia> index_to_sorted_pair(2)
(1, 3)
julia> index_to_sorted_pair(3)
(2, 3)
julia> index_to_sorted_pair(4)
(1, 4)
julia> index_to_sorted_pair(5)
(2, 4)
```
"""
@inline function index_to_sorted_pair(index)
    # pick second according to https://oeis.org/A002024 (zero-based)
    # first is the remainder
    @fastmath snd = floor(Int, (√(8 * index) + 1)/2)
    fst = index - (snd * (snd - 1)) >>> 0x1
    return (fst, snd + 1)
end

"""
    binomial3(k)

Equivalent to `binomial(k, 3)`, but more efficient.
"""
@inline binomial3(k) = k * (k - 1) * (k - 2) ÷ 6

"""
    index_to_sorted_triple(index, top)

Map combinatorial `index` to (sorted) triple of (distinct) integers. `top` must be higher or
equal to the maximum value that is expected in triple, or the behaviour of this function
is undefined.

# Example

```julia
julia> index_to_sorted_triple(1, 5)
(1, 2, 3)
julia> index_to_sorted_triple(2, 5)
(1, 2, 4)
julia> index_to_sorted_triple(3, 5)
(1, 3, 4)
julia> index_to_sorted_triple(4, 5)
(2, 3, 4)
julia> index_to_sorted_triple(5, 5)
(1, 2, 5)
```
"""
@inline function index_to_sorted_triple(index, top)
    # Perform binary search to find the last value.
    lo = 2
    hi = top
    while lo < hi - 1
        mid = lo + ((hi - lo) >> 0x1)
        go_down = binomial3(mid) < index
        lo = ifelse(go_down, mid, lo)
        hi = ifelse(go_down, hi, mid)
    end
    index -= binomial3(lo)
    fst, snd = index_to_sorted_pair(index)
    return (fst, snd, lo + 1)
end

"""
     pick_two_occupied_modes(add::BoseFS, map, chosen)

Pick two occupied modes in address `add`, indexed by `chosen`.
Return the mode indices (as `BoseFSIndex` or `FermiFSIndex`) and the number of permutations
of the selection that exist.
"""
function pick_two_occupied_modes(add::BoseFS, map, chosen)
    singlies = length(map)
    double = chosen - singlies * (singlies - 1) ÷ 2
    if double > 0
        idx, _ = pick_multiply_occupied_mode(add, map, double, 2)
        return (idx, idx), 1
    else
        fst, snd = index_to_sorted_pair(chosen)
        return (map[fst], map[snd]), 2
    end
end
function pick_two_occupied_modes(add::FermiFS, map, chosen)
    fst, snd = index_to_sorted_pair(chosen)
    return (map[fst], map[snd]), 2
end

"""
    pick_multiply_occupied_mode(add, map, chosen, num)

Pick a mode that contains at least `num` particles.
"""
@inline function pick_multiply_occupied_mode(add::BoseFS, map, chosen, num)
    idx = first(map)
    i = 1
    # Weird order in loop skips some branches.
    while true
        chosen -= idx.occnum ≥ num
        chosen == 0 && return idx, i
        i += 1
        idx = map[i]
    end
end

"""
     pick_three_occupied_modes(add::BoseFS, map, chosen)

Pick three occupied modes in address `add`, indexed by `chosen`.
Return the mode indices (as `BoseFSIndex` or `FermiFSIndex`) and the number of permutations
of the selection that exist.
"""
function pick_three_occupied_modes(add::BoseFS, map, chosen)
    singlies = length(map)
    double = chosen - singlies * (singlies - 1) * (singlies - 2) ÷ 6
    if double > 0
        # Will pick a doubly or triply occupied mode.
        doublies = count(i -> i.occnum ≥ 2, map)
        triple = double - doublies * (singlies - 1)
        if triple > 0
            # Pick a triply occupied mode.
            idx, _ = pick_multiply_occupied_mode(add, map, triple, 3)
            return (idx, idx, idx), 1
        else
            # Pick a doubly occupied mode and a singly occupied one.
            double, single = fldmod1(double, singlies - 1)
            double_idx, double = pick_multiply_occupied_mode(add, map, double, 2)
            if single ≥ double
                single_idx = map[single + 1]
                return (double_idx, double_idx, single_idx), 3
            else
                single_idx = map[single]
                return (single_idx, double_idx, double_idx), 3
            end
        end
    else
        fst, snd, trd = index_to_sorted_triple(chosen, singlies)
        return (map[fst], map[snd], map[trd]), 6
    end
end
function pick_three_occupied_modes(add::FermiFS, map, chosen)
    fst, snd, trd = index_to_sorted_triple(chosen, singlies)
    return (map[fst], map[snd], map[trd]), 6
end

"""
    pick_mode_avoiding_sources(add, map, chosen, avoid::Tuple)

Pick a mode in address `add` avoiding the elements of `avoid` for `BoseFS`, and avoiding
occupied modes for `FermiFS`.
"""
function pick_mode_avoiding_sources(add::BoseFS, map, chosen, avoid)
    chosen = _avoid(chosen, avoid)
    return find_mode(add, chosen)
end

@inline _avoid(chosen, ()::Tuple{}) = chosen
@inline _avoid(chosen, (a, as...)) = _avoid(chosen + (chosen ≥ a) * (a ∉ as), as)

# assumes source is occupied, which makes sense - ignores avoid
function pick_mode_avoiding_sources(add::FermiFS, map, chosen, avoid)
    i = 1
    idx = map[1]
    while i < length(map) # Could binary search? Probably not worth the effort.
        if chosen ≥ idx.mode
            chosen += 1
        else
            break
        end
        i += 1
        idx = map[i]
    end
    return idx
end
