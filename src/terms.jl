#TODO: move adjoints into functions
"""
    AbstractTerm

`AbstractTerm`s are building blocks of Hamiltonians and other operators. Generally,
operators will define [`terms`](@ref), which returns a sum of `AbstractTerms`.

The parameter `N` decides how many particles are involved in the excitation that defines the
term. For example, a two body term is of type `AbstractTerm{T,2}`.
"""
abstract type AbstractTerm{T,K} <: AbstractOperator{T} end

function Base.show(io::IO, op::AbstractTerm)
    print(io, nameof(typeof(op)))
end

"""
    SingleTermOperator

Wrapper over an `AbstractTerm` that equips it with a starting address and allows it to be
used as an operator.
"""
struct SingleTermOperator{A,T,O<:AbstractTerm{T}} <: Hamiltonian{A,T}
    term::O
    address::A
end
terms(op::SingleTermOperator) = op.term
starting_address(op::SingleTermOperator) = op.address
LOStructure(op::SingleTermOperator) = LOStructure(op.term)
LinearAlgebra.adjoint(op::SingleTermOperator) = SingleTermOperator(op.term', op.address)
