struct OperatorSum{
    A,T,L<:AbstractOperator{A,T},R<:AbstractOperator{A,T}
} <: AbstractOperator{A,T}
    left::L
    right::R
end

Base.show(io::IO, op::OperatorSum) = print(io, op.left, " + ", op.right)

Base.:+(left::AbstractOperator, right::AbstractOperator) = OperatorSum(left, right)

starting_address(op::OperatorSum) = starting_address(op.left)
LOStructure(op::OperatorSum) = combine_structure(LOStructure(op.left), LOStructure(op.right))
combine_structure(::IsHermitian, ::IsHermitian) = IsHermitian()
combine_structure(::AdjointKnown, ::IsHermitian) = AdjointKnown()
combine_structure(::IsHermitian, ::AdjointKnown) = AdjointKnown()
combine_structure(::AdjointKnown, ::AdjointKnown) = AdjointKnown()
combine_structure(_, _) = AdjointUnknown()

Base.adjoint(op::OperatorSum) = adjoint(op.left) + adjoint(op.right)


struct OperatorSumColumn{
    A,T,L<:AbstractColumn{A,T},R<:AbstractColumn{A,T}
} <: AbstractColumn{A,T}
    left::L
    right::R
    n_left::Int
    n_right::Int
end

Base.show(io::IO, col::OperatorSumColumn) = print(io, col.left, " + ", col.right)

function column(op::OperatorSum{A}, add::A) where {A}
    left = column(op.left, add)
    right = column(op.right, add)
    return OperatorSumColumn(left, right, num_offdiagonals(left), num_offdiagonals(right))
end
function column(op::OperatorSum{A}, add::A) where {N,A<:CompositeFS{N}}
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
        # TODO removing this throw makes things slower. Hide it under @boundscheck?
        throw(BoundsError(col, orig_chosen))
    end
end

function diagonal_element(col::OperatorSumColumn)
    return diagonal_element(col.left) + diagonal_element(col.right)
end

function is_mom_space(op::OperatorSum)
    return is_mom_space(op.left) || is_mom_space(op.right)
end
function is_real_space(op::OperatorSum)
    return is_real_space(op.left) || is_real_space(op.right)
end
