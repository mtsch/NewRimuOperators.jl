function sorted_sparse(ham)
    bsr = BasisSetRep(ham)
    perm = sortperm(bsr.basis)
    return sparse(ham)[perm, perm]
end

@testset "Rimu equivalence" begin
    @testset "HubbardMom1D" begin
        add = BoseFS((0,0,5,0,0))

        ham_rimu = HubbardMom1D(add; u=2, t=0.5)
        ham_new = Hubbard(add; u=2, t=0.5)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "HubbardMom1DEP" begin
        add = BoseFS((0,0,4,0,0))

        ham_rimu = HubbardMom1DEP(
            add; u=0.02, t=1.5, v_ho=3, dispersion=continuum_dispersion
        )
        ham_new = Hubbard(add; u=0.02, t=1.5, dispersion=continuum_dispersion) +
            HarmonicPotential(3)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "BoseHubbardMom1D2C" begin
        add1 = BoseFS2C((0,0,4,0,0), (0,0,2,0,0))
        add2 = CompositeFS(BoseFS((0,0,4,0,0)), BoseFS((0,0,2,0,0)))

        ham_rimu = BoseHubbardMom1D2C(add1, v=2.1, ua=2.3, ub=2.4, ta=3.1, tb=3)
        ham_new = Hubbard(add2, u=[2.3 2.1; 2.1 2.4], t=[3.1, 3])

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "HubbardMom1DEP (fermions)" begin
        add = FermiFS2C((0,0,1,1,1,0,0), (0,0,1,0,1,0,0))

        ham_rimu = HubbardMom1DEP(add; u=0.7, t=1.3, v_ho=0.7)
        ham_new = Hubbard(add; u=[0 0.7; 0.7 0], t=[1.3, 1.3]) + HarmonicPotential(0.7)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "HubbardReal1DEP" begin
        add = BoseFS((1,1,1,1,1,1))

        ham_rimu = HubbardReal1DEP(add; u=0.7, t=1.3, v_ho=0.7)
        ham_new = Hubbard(add; u=0.7, t=1.3, momentum=false) + HarmonicPotential(0.7)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "HubbardRealSpace" begin
        add = FermiFS2C((0,1,1,1,0), (0,1,1,0,0))

        ham_rimu = HubbardRealSpace(add)
        ham_new = Hubbard(add; momentum=false)

        @test sorted_sparse(ham_new) ≈ sorted_sparse(ham_rimu)
    end
    @testset "Three components" begin
        add = CompositeFS(
            BoseFS((0,0,1,0,0)),
            BoseFS((0,0,2,0,0)),
            FermiFS((0,0,1,0,0)),
        )

        u = [1 2 3;
             2 4 5
             3 5 0]
        t = [7, 8, 9]
        ham_rimu = HubbardRealSpace(add; t, u)
        ham_new_re = Hubbard(add; t, u, momentum=false)
        ham_new_mom = Hubbard(add; t, u)

        @test sorted_sparse(ham_rimu) == sorted_sparse(ham_new_re)
        @test eigen(Matrix(ham_new_re)).values[1] ≈ eigen(Matrix(ham_new_mom)).values[1]
    end
    @testset "More" begin
        add = CompositeFS(
            FermiFS((0,1,0,1,0)),
            BoseFS((0,0,2,0,0)),
            FermiFS((0,1,0,1,0)),
            BoseFS((0,0,2,0,0))
        )

        ham_re = HubbardRealSpace(add)
        ham_mo = Hubbard(add)

        @test eigs(sparse(ham_re; sizelim=1e6); which=:SR)[1][1] ≈
            eigs(sparse(ham_mo; sizelim=1e6); which=:SR)[1][1]
    end
    @testset "Transcorrelated no 3-body" begin
        add = FermiFS2C((0,0,1,1,1,0,0), (0,0,1,0,1,0,0))

        ham_rimu = Transcorrelated1D(add; v=1.1, t=0.9, three_body_term=false, v_ho=0.1)
        ham_new = Transcorrelated(add; u=1.1, t=0.9, three_body_term=false) +
            HarmonicPotential(0.1)

        @test sorted_sparse(ham_rimu) ≈ sorted_sparse(ham_new)
    end
    @testset "Transcorrelated with 3-body" begin
        add = FermiFS2C((0,0,0,1,1,0,0), (1,0,1,0,1,0,0))

        ham_rimu = Transcorrelated1D(add; v=1.1, t=0.9, three_body_term=true)
        ham_new = Transcorrelated(add; u=1.1, t=0.9, three_body_term=true)

        @test sorted_sparse(ham_rimu) ≈ sorted_sparse(ham_new)
    end
end
@testset "TC transpose" begin
    add = FermiFS2C((0,0,1,1,1,0,0), (0,0,1,0,1,0,0))
    ham = Transcorrelated(add; u=1.1, t=0.9, three_body_term=false)

    @test sparse(ham) ≠ sparse(ham)'
    @test sparse(ham)' == sparse(ham')
end
