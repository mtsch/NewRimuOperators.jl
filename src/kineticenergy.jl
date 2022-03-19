"""
    KineticEnergy(address, t; dispersion)

```math
\\sum_k f(k) n_k
```

where ``f`` is the `dispersion`.
"""
struct KineticEnergy{A,M,F} <: AbstractOperator{A,Float64}
    address::A
    dispersion::SVector{M,Float64}
    fun::F
end
function KineticEnergy(address, fun; dispersion=hubbard_dispersion)
    M = num_modes(address)
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    kes = SVector{M}(dispersion.(kr))
    return KineticEnergy(address, kes, fun)
end
starting_address(op::KineticEnergy) = op.address

num_offdiagonals(::KineticEnergy, _, _) = 0
function diagonal_element(op::KineticEnergy, address, map, comp=1)
    result = 0.0
    for index in map
        result += index.occnum * op.dispersion[index.mode]
    end
    return result * op.fun(comp)
end

CompositeAction(::KineticEnergy) = NoCompositeAction()
LOStructure(::KineticEnergy) = IsHermitian() # should be IsDiagonal?
