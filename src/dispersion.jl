"""
    KineticEnergy(address, t; dispersion)

```math
\\sum_k f(k) n_k
```

where ``f`` is the `dispersion`.
"""
struct KineticEnergy{A,M} <: AbstractOperator{A,Float64}
    address::A
    dispersion::SVector{M,Float64}
end
function KineticEnergy(address, t; dispersion=hubbard_dispersion)
    M = num_modes(address)
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    kes = SVector{M}(t .* dispersion.(kr))
    return KineticEnergy(address, kes)
end
starting_address(op::KineticEnergy) = op.address

num_offdiagonals(::KineticEnergy, _, _) = 0
function diagonal_element(op::KineticEnergy, _, map)
    result = 0.0
    for index in map
        result += index.occnum * op.dispersion[index.mode]
    end
    return result
end

CompositeAction(::KineticEnergy) = NoCompositeAction()
