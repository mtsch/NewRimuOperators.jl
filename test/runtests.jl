using Test
using Rimu
using LinearAlgebra
using NewRimuOperators

function sorted_sparse(ham)
    bsr = BasisSetRep(ham)
    perm = sortperm(bsr.basis)
    return sparse(ham)[perm, perm]
end
function offdiags_only(ham)
    matrix = sorted_sparse(ham)
    return matrix - diagm(diag(matrix))
end

@testset "Rimu equivalence" begin
    @testset "HubbardMom1D" begin
        add = BoseFS((0,0,5,0,0))

        ham_rimu = HubbardMom1D(add; u=2, t=0.5)
        ham_new = HubbardMom(add; u=2, t=0.5)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "HubbardMom1DEP" begin
        add = BoseFS((0,0,4,0,0))

        ham_rimu = HubbardMom1DEP(
            add; u=0.02, t=1.5, v_ho=3, dispersion=continuum_dispersion
        )
        ham_new = HubbardMom(add; u=0.02, t=1.5, dispersion=continuum_dispersion) +
            HarmonicOscillatorMom(add, 3)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "BoseHubbardMom1D2C" begin
        add1 = BoseFS2C((0,0,4,0,0), (0,0,2,0,0))
        add2 = CompositeFS(BoseFS((0,0,4,0,0)), BoseFS((0,0,2,0,0)))

        ham_rimu = BoseHubbardMom1D2C(add1, v=2.1, ua=2.1, ub=2.1, ta=3, tb=3)
        ham_new = HubbardMom(add2, u=2.1, t=3) # TODO: needs separate ua and ub

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "HubbardMom1DEP (fermions)" begin
        add = FermiFS2C((0,0,1,1,1,0,0), (0,0,1,0,1,0,0))

        ham_rimu = HubbardMom1DEP(add; u=0.7, t=1.3, v_ho=0.7)
        ham_new = HubbardMom(add; u=0.7, t=1.3) + HarmonicOscillatorMom(add, 0.7)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "Three components" begin
        add1 = BoseFS((0,0,4,0,0))
        add2 = CompositeFS(
            BoseFS((0,0,1,0,0)),
            BoseFS((0,0,2,0,0)),
            FermiFS((0,0,1,0,0)),
        )

        ham_rimu = HubbardMom1D(add1; dispersion=continuum_dispersion)
        ham_new = HubbardMom(add1; dispersion=continuum_dispersion)

        @test eigen(Matrix(ham_new)).values[1] ≈ eigen(Matrix(ham_rimu)).values[1]
    end

    @test "Transcorrelated - no 3-body" begin
        add = FermiFS2C((0,0,1,1,1,0,0), (0,0,1,0,1,0,0))

        ham_rimu = Transcorrelated1D(add; v=1.1, t=0.9, three_body_term=false, v_ho=0.1)
        ham_new = Transcorrelated1D(add; v=1.1, t=0.9, three_body_term=false, v_ho=0.1)

        @test offdiags_only(ham_rimu) ≈ offdiags_only(ham_new)
    end
    @test "Transcorrelated - with 3-body" begin
        add = FermiFS2C((0,0,0,1,1,0,0), (1,0,1,0,1,0,0))

        ham_rimu = Transcorrelated1D(add; v=1.1, t=0.9, three_body_term=true)
        ham_new = Transcorrelated1D(add; v=1.1, t=0.9, three_body_term=true)

        @test offdiags_only(ham_rimu) ≈ offdiags_only(ham_new)
    end
end
