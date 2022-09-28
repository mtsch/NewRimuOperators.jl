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
function conserves_momentum(term, add; fold=false)
    m = mom(add; fold)
    return all(BasisSetRep(SingleTermOperator(term, add)).basis) do fs
        mom(fs; fold) == m
    end
end
function matrix_fully_dense(term, add)
    matrix = Matrix(SingleTermOperator(term, add))
    return all(!iszero, matrix)
end

function same_groundstate(term, bose::BoseFS)
    fermi = Vector{Int}[]
    for idx in OccupiedModeMap(bose)
        for _ in 1:idx.occnum
            push!(fermi, Int[i == idx.mode for i in 1:num_modes(bose)])
        end
    end
    fermi = CompositeFS(map(FermiFS, fermi)...)

    @test eigvals(Matrix(SingleTermOperator(term, bose)))[1] ≈
        eigvals(Matrix(SingleTermOperator(term, fermi)))[1]

    # Make a mixed one as well
    if num_particles(bose) > 3
        fermi = Vector{Int}[]
        occs_to_move = 3
        new_bose = zeros(Int, num_modes(bose))

        for idx in OccupiedModeMap(bose)
            for _ in 1:idx.occnum
                if occs_to_move > 0
                    new_bose[idx.mode] += 1
                    occs_to_move -= 1
                else
                    push!(fermi, Int[i == idx.mode for i in 1:num_modes(bose)])
                end
            end
        end

        mixture = CompositeFS(BoseFS(new_bose), map(FermiFS, fermi)...)

        @test eigvals(Matrix(SingleTermOperator(term, bose)))[1] ≈
            eigvals(Matrix(SingleTermOperator(term, mixture)))[1]
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

            same_groundstate(ParticleCountTerm(ConstFunction(1)), BoseFS((1,1,1,2)))
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

            same_groundstate(NeighbourOneBodyTerm(ConstFunction(1)), BoseFS((1,1,1,2)))
        end

        @testset "FullOneBodyTerm" begin
            add = CompositeFS(
                BoseFS((0, 0, 3, 0)),
                FermiFS((1, 1, 0, 0)),
            )
            term_real = FullOneBodyTerm((σ, q, p) -> σ == 1 ? (p - q) : (p * q))
            @test num_offdiagonals(term_real, add) == 3 * 3
            @test diagonal_element(term_real, add) == 5
            @test !conserves_momentum(term_real, add)
            check_lo_structure(term_real, add, AdjointKnown())

            term_complex = FullOneBodyTerm((σ, q, p) -> σ == 1 ? (p - q * im) : (p * q + im))
            @test num_offdiagonals(term_complex, add) == 3 * 3
            @test diagonal_element(term_complex, add) == 14 - 7im
            check_lo_structure(term_complex, add, AdjointKnown())

            @test matrix_fully_dense(
                FullOneBodyTerm(ConstFunction(1)), FermiFS((0,0,1,0,0,0))
            )
            @test matrix_fully_dense(
                FullOneBodyTerm(ConstFunction(1)), BoseFS((1,0,0,0,0,0,0))
            )
            @test !matrix_fully_dense(
                FullOneBodyTerm(ConstFunction(1)), BoseFS((0,2,0,0,0,0))
            )

            # TODO check again - if complex weird stuff happens
            same_groundstate(FullOneBodyTerm((_, p, q) -> (p + q)), BoseFS((1,3,1,0)))
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

            same_groundstate(OnsiteInteractionTerm(ConstFunction(1)), BoseFS((1,5,1)))
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
            @test !conserves_momentum(term_real_fold, add; fold=false)
            @test conserves_momentum(term_real_fold, add; fold=true)

            term_complex_fold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) + q * im
            )
            check_lo_structure(term_complex_fold, add, AdjointKnown())

            term_real_nofold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) * p; fold=false
            )
            @test conserves_momentum(term_real_nofold, add; fold=false)
            @test conserves_momentum(term_real_nofold, add; fold=true)
            check_lo_structure(term_real_nofold, add, AdjointKnown())

            term_complex_nofold = MomentumTwoBodyTerm(
                (σ,τ,p,q,r,s) -> σ * τ * (p + q) + (r + s) * (p - s) + p * im; fold=false
            )
            check_lo_structure(term_complex_nofold, add, AdjointKnown())

            @test matrix_fully_dense(
                MomentumTwoBodyTerm(ConstFunction(1)), FermiFS2C((0,1,0,0), (0,0,1,0))
            )
            @test matrix_fully_dense(
                MomentumTwoBodyTerm(ConstFunction(1)), BoseFS((0,0,0,0,2,0))
            )
            @test !matrix_fully_dense(
                MomentumTwoBodyTerm(ConstFunction(1)), BoseFS((0,2,0,0,0,1))
            )

            same_groundstate(
                MomentumTwoBodyTerm((_,_,p,q,r,s) -> (s - p)^2), BoseFS((1,5,1))
            )
        end

        @testset "FullTwoBodyTerm" begin
            add = CompositeFS(
                BoseFS((0, 0, 2, 0, 1, 0)),
                FermiFS((1, 1, 0, 0, 0, 0)),
            )
            term_real = FullTwoBodyTerm((σ,τ,p,q,r,s) -> σ * τ * (p + q + r + s) + p)
            check_lo_structure(term_real, add, AdjointKnown())
            @test !conserves_momentum(term_real, add; fold=true)

            term_complex = FullTwoBodyTerm((σ,τ,p,q,r,s) -> σ * τ * (p + q + r + s) + p*im)
            check_lo_structure(term_complex, add, AdjointKnown())

            @test matrix_fully_dense(
                FullTwoBodyTerm(ConstFunction(1)), FermiFS2C((0,1,0,0), (0,0,1,0))
            )
            @test matrix_fully_dense(
                FullTwoBodyTerm(ConstFunction(1)), BoseFS((0,0,0,0,2,0))
            )
            @test !matrix_fully_dense(
                FullTwoBodyTerm(ConstFunction(1)), BoseFS((0,2,0,0,0,1))
            )

            # TODO
            same_groundstate(
                FullTwoBodyTerm(ConstFunction(1)), BoseFS((1,5,1))
            )
            same_groundstate(
                FullTwoBodyTerm((_,_,p,q,r,s) -> (s - p)^2), BoseFS((1,5,1))
            )
        end
    end

    @testset "MomentumThreeBodyTerm" begin
        add = CompositeFS(
            BoseFS((0, 0, 0, 3, 0)),
            FermiFS((1, 1, 0, 0, 0)),
            FermiFS((1, 1, 0, 0, 0)),
        )
        func(σ,τ,υ,p,q,r,s,t,u) = σ * τ * υ * (p - u) * (q - t) * s

        term = MomentumThreeBodyTerm(func)
        check_lo_structure(term, add, AdjointKnown())
        @test conserves_momentum(term, add; fold=false)

        # Need more complicated function here. For fermions, ConstFunction(1) cancels itself
        # out.
        func(σ, τ, p, q, r, s, t, u) = (p - u) * (q - t)

        @test matrix_fully_dense(
            MomentumThreeBodyTerm(func), FermiFS2C((0,1,0,0,0,0), (0,0,1,1,0,0))
        )
        @test matrix_fully_dense(
            MomentumThreeBodyTerm(func), FermiFS2C((0,1,1,0,0,0), (0,0,1,0,0,0))
        )
        @test !matrix_fully_dense(
            MomentumThreeBodyTerm(func), FermiFS2C((1,0,1,0,0,0), (0,0,1,0,0,1))
        )

        # TODO
        same_groundstate(MomentumThreeBodyTerm(ConstFunction(1)), BoseFS((1,5,1)))
    end
end
