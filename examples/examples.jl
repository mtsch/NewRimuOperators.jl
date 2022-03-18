using Rimu, NewRimuOperators, LinearAlgebra

# Sanity check with small system:
add = BoseFS((0,0,3,0,0))
H1 = HubbardMom1D(add; u=6.0)
H2 = HubbardMom(add; u=6.0)

Matrix(H1) == Matrix(H2)
# true

# Try with a complicated fock state:
add = CompositeFS(
    FermiFS((0,1,0,0)),
    BoseFS((0,2,0,0)),
    FermiFS((0,1,0,0))
)
# Note: currently no way to set different parameters values for different interactions.
Hc = HubbardMom(add; v_ho=1)
typeof(Hc)
# NewRimuOperators.OperatorSum{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, NewRimuOperators.OperatorSum{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, KineticEnergy{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, 4}, MomentumTransfer{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, ConstFunction{Float64}}}, HarmonicOscillatorMom{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, 4}}
# Yikes

offdiagonals(Hc, add)
# Seems reasonable

eigen(Matrix(Hc)).values[1]
# -2.562060462618259

typeof(column(Hc, add))
# NewRimuOperators.OperatorSumColumn{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, NewRimuOperators.OperatorSumColumn{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, NewRimuOperators.CompositeColumn{3, CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, KineticEnergy{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, 4}, Tuple{OccupiedModeMap{1, FermiFSIndex}, OccupiedModeMap{2, BoseFSIndex}, OccupiedModeMap{1, FermiFSIndex}}}, NewRimuOperators.CompositeColumn{3, CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, MomentumTransfer{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, ConstFunction{Float64}}, Tuple{OccupiedModeMap{1, FermiFSIndex}, OccupiedModeMap{2, BoseFSIndex}, OccupiedModeMap{1, FermiFSIndex}}}}, NewRimuOperators.CompositeColumn{3, CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, Float64, HarmonicOscillatorMom{CompositeFS{3, 4, 4, Tuple{FermiFS{1, 4, BitString{4, 1, UInt8}}, BoseFS{2, 4, BitString{5, 1, UInt8}}, FermiFS{1, 4, BitString{4, 1, UInt8}}}}, 4}, Tuple{OccupiedModeMap{1, FermiFSIndex}, OccupiedModeMap{2, BoseFSIndex}, OccupiedModeMap{1, FermiFSIndex}}}}
# Yikes

# Some benchmarks
using BenchmarkTools

add = BoseFS((0,0,0,0,10,0,0,0,0,0))
H1 = HubbardMom1D(add)
H2 = HubbardMom(add)

s_strat = DoubleLogUpdate(targetwalkers=10000)
@benchmark lomc!(H1; style=IsDynamicSemistochastic(), s_strat, laststep=10_000)
@benchmark lomc!(H2; style=IsDynamicSemistochastic(), s_strat, laststep=10_000)

add1 = BoseFS((0,0,4,0,0,0))
add2 = CompositeFS(
    BoseFS((0,0,2,0,0,0)),
    BoseFS((0,0,2,0,0,0)),
    FermiFS((0,0,1,0,0,0)),
    FermiFS((0,0,1,0,0,0)),
)

ham_rimu = HubbardMom1D(add1; dispersion=continuum_dispersion)
ham_new = HubbardMom(add2; dispersion=continuum_dispersion)

s_strat = DoubleLogUpdate(targetwalkers=100000)
laststep = 10000
style = IsDynamicSemistochastic()

@benchmark lomc!(ham_rimu; style, s_strat, laststep)
@benchmark lomc!(ham_new; style, s_strat, laststep) # <- slower
