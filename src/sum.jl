"""


"""
struct OperatorSum{T,L<:AbstractOperator,R<:AbstractOperator} <: AbstractOperator{T}
    left::L
    right::R
end
function OperatorSum(left::L, right::R) where {L,R}
    T = promote_type(eltype(left), eltype(right))
    return OperatorSum{T,L,R}(left, right)
end

Base.show(io::IO, op::OperatorSum) = print(io, op.left, " + ", op.right)


LOStructure(op::OperatorSum) = combine_structure(LOStructure(op.left), LOStructure(op.right))
combine_structure(::IsHermitian, ::IsHermitian) = IsHermitian()
combine_structure(::AdjointKnown, ::IsHermitian) = AdjointKnown()
combine_structure(::IsHermitian, ::AdjointKnown) = AdjointKnown()
combine_structure(::AdjointKnown, ::AdjointKnown) = AdjointKnown()
combine_structure(_, _) = AdjointUnknown()

Base.adjoint(op::OperatorSum) = adjoint(op.left) + adjoint(op.right)

# Default cases.
Base.:+(left::AbstractOperator, right::AbstractOperator) = OperatorSum(left, right)

###
### Columns
###
struct OperatorSumColumn{
    A,T,L<:AbstractColumn{A,T},R<:AbstractColumn{A,T}
} <: AbstractColumn{A,T}
    left::L
    right::R
    n_left::Int
    n_right::Int
end

Base.show(io::IO, col::OperatorSumColumn) = print(io, col.left, " + ", col.right)

function column(op::OperatorSum, add::A) where {A}
    left = column(op.left, add)
    right = column(op.right, add)
    return OperatorSumColumn(left, right, num_offdiagonals(left), num_offdiagonals(right))
end
function column(op::OperatorSum, add::A) where {N,A<:CompositeFS{N}}
    left = column(op.left, add)
    right = column(op.right, add)
    return OperatorSumColumn(left, right, num_offdiagonals(left), num_offdiagonals(right))
end

num_offdiagonals(col::OperatorSumColumn) = col.n_left + col.n_right
function get_offdiagonal(col::OperatorSumColumn, chosen)
    orig_chosen = chosen
    if chosen ≤ col.n_left
        return get_offdiagonal(col.left, chosen)
    elseif chosen ≤ col.n_left + col.n_right
        chosen -= col.n_left
        return get_offdiagonal(col.right, chosen)
    else
        # TODO removing this throw makes things faster. Hide it under @boundscheck?
        throw(BoundsError(col, orig_chosen))
    end
end

function diagonal_element(col::OperatorSumColumn)
    return diagonal_element(col.left) + diagonal_element(col.right)
end
