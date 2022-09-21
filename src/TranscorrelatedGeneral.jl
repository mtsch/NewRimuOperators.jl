using Rimu.Hamiltonians: n_to_k, correlation_factor

struct OneBodyJastrowFunction
end

struct TwoBodyJastrowFunction
end

struct QTensor
    values
end
function QTensor()

end
(Q::QTensor)(m,n) = Q.values[m,n]

struct WTensor
    values
end
function WTensor()
    
end
(w::WTensor)(n) = w.values[abs(n) + 1]

# Redundant; see KineticEnergyFunction
struct SecondDerivativeFunction
    U::SVector
    cf::TwoBodyJastrowFunction
end
function SecondDerivativeFunction(u, cutoff)
    return SecondDerivativeFunction(diag(u), TwoBodyJastrowFunction(M, cutoff))
end
function (f::SecondDerivativeFunction)()
    
end

struct NonHermitianFunction
    J::SVector
    cf::TwoBodyJastrowFunction
end
function NonHermitianFunction(j, cutoff)
    return NonHermitianFunction(diag(j), TwoBodyJastrowFunction(M, cutoff))
end
function (f::NonHermitianFunction)()
end

struct DerivativeSquaredFunction
    P::SMatrix
    W::WTensor
end
function DerivativeSquaredFunction(u, j, M, cutoff)
    P = u.^2 ./ j / M
    P = 2P - Diagonal(diag(P))    # double the offdiagonal terms
    return DerivativeSquaredFunction(P, WTensor(M, cutoff))
end
function (f::DerivativeSquaredFunction)(_, _, b, a, p2, q2, q1, p1)
    k = p1 - p2
    @assert k == q2 - q1    # is this already checked?
    return f.P[a,b] * f.W(k)
end

struct ThreeBodyFunction
    P::SArray
    J::SVector
    U::SVector
    Q::QTensor
end
function ThreeBodyFunction(u, j, M, cutoff)
    P = u ./ j / M
    return ThreeBodyFunction(P, diag(j), diag(u), QTensor(M, cutoff))
end
function (f::ThreeBodyFunction)(a2, b2, c2, c1, b1, a1, p2, q2, r2, r1, q1, p1)
    if all([a1,b1,b2] .== [a2,b2,c2])
        if a1 == b1 == c1
            # intracomponent
            k1 = p1 - p2
            k2 = q2 - q1
            @assert r2 == r1 + k1 - k2
            Pval = f.P[a1,a1]^2
        elseif c1 == b1
            # intercomponent A
            k1 = p1 - p2
            k2 = q2 - q1
            @assert r2 == r1 + k1 - k2
            Pval = f.P[a1,b1]^2 * f.J[a1]
        elseif a1 == b1
            # intercomponent B
            k1 = p1 - p2
            k2 = q2 - q1
            @assert r2 == r1 + k1 - k2
            Pval = f.P[a1,b1]^2 * f.J[a1]
        elseif a1 ≠ b1 ≠ c1
            # three component
            loser, a, b = work_out_which_component_loses_momentum(p2, q2, r2, r1, q1, p1)
            Pval *= 2f.J[loser] * f.P[a,loser] * f.P[b,loser]
            if loser == a1                

            elseif loser == b1

            elseif loser == c1

            else
                throw(ErrorException("Bad."))
            end
        end
        Qval = f.Q(k1,k2)
    else
        # mixed components
        Pval, Qval = black_magic()
    end
    return Pval * Qval
end

# Currently only implemented for 1D lattice
get_dimension_from_address_somehow_probably_need_to_use_lattice_geometry(_) = 1

"""
    TranscorrelatedGeneral1D(address; kwargs...)

Implements a transcorrelated Hamiltonian for any number of components with arbitrary masses 
on a lattice using momentum space formalism. Assumes that the interparticle interactions 
are contact interactions represented by Bethe-Peierls boundary conditions.

This is a generalisation of [Jeszenski *et al.* (2018)](https://arxiv.org/abs/1806.11268) 
and [Jeszenski *et al.* (2020)](https://arxiv.org/abs/2002.05987).

```math
\\begin{aligned}
\\tilde{H} - H &= 
\\end{aligned}
```

# Arguments

* `address`: The starting address, defines number of particles and sites.
* `u`: A symmetric matrix of interaction strengths between particles. Diagonal (offdiagonal) 
elements set intracomponent (intercomponent) interaction strengths. 
* `masses`: The masses of each component, in units of an arbitrary mass ratio ``m_0``.
* `BHJs`: Alternatively, set the Bose-Hubbard parameters `J` explicitly. Overrides `masses`.
* `dim`: Set the dimension of the system. Defaults to 1D.
* `cutoff`: integer `n` that sets the cutoff momentum ``k_c = 2\\pi n / M``
* `threebodyQ`: flag to enable or disable three-body terms

# See also

* [`Transcorrelated1D`](@ref)
"""
struct TranscorrelatedGeneral{A,C,D,O<:AbstractOperator{Float64}} <: Hamiltonian{A,Float64}
    address::A
    u::SMatrix{C,C,Float64} # interparticle interaction parameters, inter- and intracomponent
    j::SVector{C,Float64} # energy scales; sets component masses
    cutoff::Int
    terms::O
    threebody::Bool
end

function TranscorrelatedGeneral(add::CompositeFS{C}; 
    u = ones(C,C), 
    BHJs = nothing, 
    masses = ones(C), 
    cutoff = 1,
    threebody = false,
    fold = false
    ) where {C}

    M = num_modes(add)
    D = get_dimension_from_address_somehow_probably_need_to_use_lattice_geometry(address)

    if isnothing(BHJs)
        @assert length(masses) == C
        jdiag = float(masses .* M^2/2)
    else
        @assert length(BHJs) == C
        jdiag = float(BHJs)
    end
    j = Diagonal(jdiag)
    for a in 1:C, b in a+1:C
        j[a,b] = (j[a] + j[b])/2
        j[b,a] = j[a,b]
    end

    # bare hamiltonian
    ke = ParticleCountTerm(KineticEnergyFunction(add, j, continuum_dispersion))
    int = MomentumTwoBodyTerm(u ./ M; fold)
    bare = ke + int

    # transcorrelated terms
    d2dx2 = MomentumTwoBodyTerm(KineticEnergyFunction(add, u/M, continuum_dispersion))
    non_herm = MomentumTwoBodyTerm(NonHermitianFunction(j, cf); fold)
    ddx_sq = MomentumTwoBodyTerm(DerivativeSquaredFunction(u, j, M, cutoff); fold)

    terms = bare + ke + d2dx2 + non_herm + ddx_sq

    if threebody
        terms += MomentumThreeBodyTerm(ThreeBodyFunction(u, j, M, cutoff); fold)
    end

    return TranscorrelatedGeneral{typeof(add),C,D,typeof(terms)}(add, u, j, cutoff, terms, threebody)
end

function Base.show(io::IO, h::TranscorrelatedGeneral)
    print(io, "TranscorrelatedGeneral(")
    print(IOContext(io, :compact => true), h.address)
    print(io, ", u=$(h.u), t=$(h.t), cutoff=$(h.cutoff), threebody=$(h.three_body_term)")
    print(io, ")")
end

starting_address(h::TranscorrelatedGeneral) = h.address
LOStructure(::Type{<:TranscorrelatedGeneral}) = AdjointKnown()
function Base.adjoint(h::TranscorrelatedGeneral)
    return TranscorrelatedGeneral(h.address, h.u, h.t, h.cutoff, adjoint(h.terms), h.three_body_term)
end
terms(h::TranscorrelatedGeneral) = h.terms
basis(h::TranscorrelatedGeneral) = h.basis