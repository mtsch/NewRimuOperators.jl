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
        @test !ishermitian(mat)
        @test mat' == Matrix(op')
    else
        @test_throws ArgumentError op'
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
                BoseFS((0, 0, 3, 0)),
                FermiFS((1, 1, 0, 0)),
            )
            term = MomentumTwoBodyTerm((σ,τ,p,q,k) -> σ * τ * mod1(k,num_modes(add)) + (p - q))
        end

        @testset "FullTwoBodyTerm" begin
            add = CompositeFS(
                FermiFS((0, 0, 1, 0, 1, 0)),
                FermiFS((1, 1, 0, 0, 0, 0)),
            )
            term = FullTwoBodyTerm((σ,τ,p,q,r,s) -> σ * τ * (p + q + r + s))
        end
    end

    @testset "MomentumThreeBodyTerm" begin
        add = CompositeFS(
            FermiFS((0, 0, 1, 0, 1, 0)),
            FermiFS((1, 1, 0, 0, 0, 0)),
        )
        term = MomentumThreeBodyTerm((σ,τ,k,l) -> σ * τ * (k + l))
    end
end
