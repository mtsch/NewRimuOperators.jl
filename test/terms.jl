using LinearAlgebra
using NewRimuOperators
using Rimu
using Test

using NewRimuOperators:
    ParticleCountTerm, OnsiteInteractionTerm, NeighbourOneBodyTerm, FullOneBodyTerm,
    MomentumTwoBodyTerm, FullTwoBodyTerm, MomentumThreeBodyTerm

@testset "terms" begin
    function one_component_changed_or_zero(term, add)
        # Only one address changes at a time
        for (new_add, val) in offdiagonals(term, add)
            if new_add.components[1] ≠ add.components[1]
                new_add.components[2] == add.components[2] || return false
            elseif new_add.components[2] ≠ add.components[2]
                new_add.components[1] == add.components[1] || return false
            else
                val == 0 || return false
            end
        end
        return true
    end

    @testset "ParticleCountTerm" begin
        add = CompositeFS(
            BoseFS((0, 1, 2, 0)),
            FermiFS((1, 1, 0, 0)),
        )
        term = ParticleCountTerm((σ, p) -> σ * p)
        @test length(offdiagonals(term, add)) == 0
        @test diagonal_element(term, add) == 2 + 6 + 2 * (1 + 2)
    end

    @testset "OnsiteInteractionTerm" begin
        add = CompositeFS(
            BoseFS((0, 1, 3, 0)),
            FermiFS((1, 1, 1, 0)),
        )
        term = OnsiteInteractionTerm((σ, τ) -> σ == τ ? 10 : 2)
        @test length(offdiagonals(term, add)) == 0
        @test diagonal_element(term, add) == 30 + 2 * (1 + 3)
    end

    @testset "NeighbourOneBodyTerm" begin
        add = CompositeFS(
            BoseFS((0, 0, 3, 0)),
            FermiFS((1, 1, 1, 0)),
        )
        term = NeighbourOneBodyTerm(identity)
        @test length(offdiagonals(term, add)) == 2 + 6
        @test one_component_changed_or_zero(term, add)
        @test diagonal_element(term, add) == 0
    end

    @testset "FullOneBodyTerm" begin
        add = CompositeFS(
            BoseFS((0, 0, 3, 0)),
            FermiFS((1, 1, 0, 0)),
        )
        term = FullOneBodyTerm((σ, q, p) -> σ == 1 ? (p - q * im) : (p * q + im))
        @test num_offdiagonals(term, add) == 3 * 3
        @test diagonal_element(term, add) == 14 - 7im
        @test one_component_changed_or_zero(term, add)
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

    @testset "MomentumThreeBodyTerm" begin
        add = CompositeFS(
            FermiFS((0, 0, 1, 0, 1, 0)),
            FermiFS((1, 1, 0, 0, 0, 0)),
        )
        term = MomentumThreeBodyTerm((σ,τ,k,l) -> σ * τ * (k + l))
    end
end
