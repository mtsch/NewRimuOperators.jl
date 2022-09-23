# WFunction: NTuple? faster for small M. For QTensor, benchmark matrix vs compute on the fly, with precomputed correlation function
# custom correlation functions: precompute u(k), and if possible supply closed form for W and Q, else compute them from general formulae.
# wrappers over SArrays? I don't need this since my parameters will always be an array indexed by components
# DONE check consistency of k from all input momenta -  already done in terms? No, I should calculate k from input p1,q1,q2,p2
# DONE double counting factor of 2 already handled
# combine all three-body terms into a single sum over a,b,c
# Eq. 37: second term should be babbab or abbbba ?

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
    values::NTuple
end
function WTensor()
    
end
(w::WTensor)(n) = w.values[abs(n) + 1]

# Similar to `KineticEnergyFunction`
struct SecondDerivativeFunction
    U::SVector
    cf::TwoBodyJastrowFunction
end
function SecondDerivativeFunction(u, M, cutoff)
    return SecondDerivativeFunction(diag(u)/M, TwoBodyJastrowFunction(M, cutoff))
end
function (f::SecondDerivativeFunction)(a, b, p1, q1, q2, p2)
    k = p1 - p2
    @assert k == q2 - q1
    return 2 * f.U[a,b] * n_to_k(k)^2 * f.cf[k]
end

struct NonHermitianFunction
    U::SMatrix
    J::SMatrix
    cf::TwoBodyJastrowFunction
end
function NonHermitianFunction(u, j, M, cutoff)
    return NonHermitianFunction(u/M, j, TwoBodyJastrowFunction(M, cutoff))
end
function (f::NonHermitianFunction)(a, b, p1, q1, q2, p2)
    U = f.U
    J = f.J
    k = p1 - p2
    @assert k == q2 - q1
    val = 2 * U[a,b] * n_to_k(k) * (J[a,a] * n_to_k(p1) - J[b,b] * n_to_k(q1)) / J[a,b] * f.cf[k]
    return val
end

struct DerivativeSquaredFunction
    P::SMatrix
    W::WTensor
end
function DerivativeSquaredFunction(u, j, M, cutoff)
    P = 2u.^2 ./ j / M  # double counting of components handled by terms
    return DerivativeSquaredFunction(P, WTensor(M, cutoff))
end
function (f::DerivativeSquaredFunction)(a, b, p1, q1, q2, p2)
    k = p1 - p2
    @assert k == q2 - q1
    return f.P[a,b] * f.W(k)
end

struct ThreeBodyFunction
    P::SArray
    Q::QTensor
end
function ThreeBodyFunction(u, j, M, cutoff)
    C = size(u)[1]
    P = zeros(C,C,C)
    # combine u and j into one mega array
    
    return ThreeBodyFunction(SArray{Tuple{C,C,C}}(P), QTensor(M, cutoff))
end
function (f::ThreeBodyFunction)(a2, b2, c2, c1, b1, a1, p1, q1, r1, r2, q2, p2) # too many component indices
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
* `u`: A symmetric matrix of interaction strengths between particles, corresponding to the 
    Bose-Hubbard parameter `U`. Diagonal (offdiagonal) elements set intracomponent 
    (intercomponent) interaction strengths. 
* `masses`: The masses of each component, in units of an arbitrary mass ratio ``m_0``.
* `BHJs`: Alternatively, set the Bose-Hubbard parameters `J` explicitly. Overrides `masses`.
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
    int = MomentumTwoBodyTerm(u / M; fold)
    bare = ke + int

    # transcorrelated terms - combine these three functions
    d2dx2 = MomentumTwoBodyTerm(SecondDerivativeFunction(u, M, cutoff))
    non_herm = MomentumTwoBodyTerm(NonHermitianFunction(u, j, M, cutoff); fold)
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
    print(io, ", u=$(h.u), t=$(h.t), cutoff=$(h.cutoff), threebody=$(h.threebody)")
    print(io, ")")
end

starting_address(h::TranscorrelatedGeneral) = h.address
LOStructure(::Type{<:TranscorrelatedGeneral}) = AdjointKnown()
function Base.adjoint(h::TranscorrelatedGeneral)
    return TranscorrelatedGeneral(h.address, h.u, h.t, h.cutoff, adjoint(h.terms), h.threebody)
end
terms(h::TranscorrelatedGeneral) = h.terms
basis(h::TranscorrelatedGeneral) = MomentumSpace()