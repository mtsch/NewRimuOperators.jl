using LinearAlgebra
using NewRimuOperators
using Rimu
using Test

using NewRimuOperators:
    ParticleCountTerm, OnsiteInteractionTerm, NeighbourOneBodyTerm, FullOneBodyTerm,
    MomentumTwoBodyTerm, FullTwoBodyTerm, MomentumThreeBodyTerm

function check_lo_structure(term, add, structure)
    @test LOStructure(term) === structure
    op = SingleTermOperator(term, add)
    mat = Matrix(op)

    if LOStructure(op) === IsDiagonal()
        @test length(offdiagonals(term, add)) == 0
        @test isdiag(mat)
    elseif LOStructure(op) === IsHermitian()
        @test !isdiag(mat)
        @test ishermitian(mat)
    elseif LOStructure(op) === AdjointKnown()
        # This check is here to make sure the test is actually testing adjoints
        @test !ishermitian(mat)
        @test mat' == Matrix(op')
    else
        @test_throws ArgumentError op'
    end
end

function mom(add::SingleComponentFockAddress; fold=false)
    M = num_modes(add)
    res = 0
    for i in OccupiedModeMap(add)
        res += (i.mode - cld(M, 2)) * i.occnum
    end
    if fold
        return mod(res, M)
    else
        return res
    end
end
function mom(add::CompositeFS; fold=false)
    M = num_modes(add)
    res = sum(add.components) do fs
        mom(fs; fold=false)
    end
    if fold
        return mod(res, M)
    else
        return res
    end
end
function preserves_momentum(term, add; fold=false)
    m = mom(add; fold)
    return all(BasisSetRep(SingleTermOperator(term, add)).basis) do fs
        mom(fs; fold) == m
    end
end

@testset "terms" begin
    @testset "one-body" begin
        @testset "ParticleCountTerm" begin
            add = CompositeFS(
                BoseFS((0, 1, 2, 0)),
                FermiFS((1, 1, 0, 0)),
            )
            term_real = ParticleCountTerm((σ, p) -> σ * p)
            @test diagonal_element(term_real, add) == 2 + 6 + 2 * (1 + 2)
            check_lo_structure(term_real, add, IsDiagonal())

            term_complex = ParticleCountTerm((σ, p) -> σ * p * im)
            check_lo_structure(term_complex, add, AdjointKnown())
        end

        @testset "NeighbourOneBodyTerm" begin
            add = CompositeFS(
                BoseFS((0, 0, 3, 0)),
                FermiFS((1, 1, 1, 0)),
            )
            term_real = NeighbourOneBodyTerm(identity)
            @test length(offdiagonals(term_real, add)) == 2 + 6
            check_lo_structure(term_real, add, IsHermitian())
            @test all(iszero, diag(Matrix(SingleTermOperator(term_real, add))))

            term_complex = NeighbourOneBodyTerm(x -> im)
            @test eltype(term_complex) <: Complex
            check_lo_structure(term_complex, add, AdjointKnown())
        end

        @testset "FullOneBodyTerm" begin
            add = CompositeFS(
                BoseFS((0, 0, 3, 0)),
                FermiFS((1, 1, 0, 0)),
            )
            term_real = FullOneBodyTerm((σ, q, p) -> σ == 1 ? (p - q) : (p * q))
            @test num_offdiagonals(term_real, add) == 3 * 3
            @test diagonal_element(term_real, add) == 5
            @test !preserves_momentum(term_real, add)
            check_lo_structure(term_real, add, AdjointKnown())

            term_complex = FullOneBodyTerm((σ, q, p) -> σ == 1 ? (p - q * im) : (p * q + im))
            @test num_offdiagonals(term_complex, add) == 3 * 3
            @test diagonal_element(term_complex, add) == 14 - 7im
            check_lo_structure(term_complex, add, AdjointKnown())
        end
    end

    @testset "two-body" begin
        @testset "OnsiteInteractionTerm" begin
            add = CompositeFS(
                BoseFS((0, 1, 3, 0)),
                FermiFS((1, 1, 1, 0)),
            )
            term_real = OnsiteInteractionTerm((σ, τ) -> σ == τ ? 10 : 2)
            @test diagonal_element(term_real, add) == 30 + 2 * (1 + 3)
            check_lo_structure(term_real, add, IsDiagonal())

            term_complex = OnsiteInteractionTerm((σ, τ) -> σ == τ ? 10im : 2im)
            check_lo_structure(term_complex, add, AdjointKnown())
        end

        @testset "MomentumTwoBodyTerm" begin
            add = CompositeFS(
                BoseFS((0, 1, 3, 0)),
                FermiFS((1, 1, 0, 0)),
            )
            term_real_fold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) + p
            )
            check_lo_structure(term_real_fold, add, AdjointKnown())
            @test !preserves_momentum(term_real_fold, add; fold=false)
            @test preserves_momentum(term_real_fold, add; fold=true)

            term_complex_fold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) + q * im
            )
            check_lo_structure(term_complex_fold, add, AdjointKnown())

            term_real_nofold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) * p; fold=false
            )
            @test preserves_momentum(term_real_nofold, add; fold=false)
            @test preserves_momentum(term_real_nofold, add; fold=true)
            check_lo_structure(term_real_nofold, add, AdjointKnown())

            term_complex_nofold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) + p * im; fold=false
            )
            check_lo_structure(term_complex_nofold, add, AdjointKnown())
        end

        @testset "FullTwoBodyTerm" begin
            add = CompositeFS(
                FermiFS((0, 0, 1, 0, 1, 0)),
                FermiFS((1, 1, 0, 0, 0, 0)),
            )
            term_real = FullTwoBodyTerm((σ,τ,p,q,r,s) -> σ * τ * (p + q + r + s) + p)
            check_lo_structure(term_real, add, AdjointKnown())
            @test !preserves_momentum(term_real, add; fold=true)

            term_complex = FullTwoBodyTerm((σ,τ,p,q,r,s) -> σ * τ * (p + q + r + s) + p*im)
            check_lo_structure(term_complex, add, AdjointKnown())
        end
    end

    @testset "MomentumThreeBodyTerm" begin
        add = CompositeFS(
            FermiFS((0, 0, 1, 0, 1, 0)),
            FermiFS((1, 1, 0, 0, 0, 0)),
        )
        term = MomentumThreeBodyTerm((σ,τ,p,q,r,s,t,u) -> σ * τ * (p - u) * (q - t) * s)
        check_lo_structure(term, add, AdjointKnown())
        @test preserves_momentum(term, add; fold=false)
    end
end
