struct ExtendedHamiltonian{A,T,E<:Tuple,H<:Hamiltonian{A,T},O} <: Hamiltonian{A,T}
    hamiltonian::H
    extensions::E
    terms::O
end

terms(ext::ExtendedHamiltonian) = ext.terms
basis(ext::ExtendedHamiltonian) = basis(ext.hamiltonian)
LOStructure(ext::ExtendedHamiltonian) = LOStructure(ext.terms) # TODO?
starting_address(ext::ExtendedHamiltonian) = starting_address(ext.hamiltonian)
# TODO: add marker for adjointness
Base.adjoint(ext::ExtendedHamiltonian) = ExtendedHamiltonian(ext.hamiltonian, ext.extensions, ext.terms')

function Base.show(io::IO, ext::ExtendedHamiltonian)
    print(io, ext.hamiltonian)
    for e in ext.extensions
        print(io, " + ", e)
    end
end

"""
    abstract type ExtensionPrototype

A prototype for a potential (or other modifier that adds terms to existing operator). When
added to an operator, it can use information from it to construct appropriate terms.

# See also

* [`HarmonicPotential`](@ref)
* [`DeltaFunctionPotential`](@ref)
* [`TranscorrelatedDeltaPotential`](@ref)
"""
abstract type ExtensionPrototype end

"""
    initialize(prototype::AbstractOperator, operator::ExtensionPrototype)

Use the information in `prototype` and `operator` to construct an appropriate term to be
added to the operator.
"""
initialize

function Base.:+(ham::Hamiltonian, proto::ExtensionPrototype)
    term = initialize(proto, ham)
    return ExtendedHamiltonian(ham, (proto,), terms(ham) + term)
end
Base.:+(proto::ExtendedHamiltonian, ham::Hamiltonian) = ham + proto

function Base.:+(ext_ham::ExtendedHamiltonian, proto::ExtensionPrototype)
    base = ext_ham.hamiltonian
    extensions = ext_ham.extensions
    term = initialize(proto, base)

    return ExtendedHamiltonian(base, (extensions..., proto), terms(ext_ham) + term)
end
