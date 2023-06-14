function apply!(state::Ket, indices, operation::Operator)
    op = basis(state)==basis(operation) ? operation : embed(basis(state), indices, operation)
    state.data = (op*state).data
    state
end

function apply!(state::Operator, indices, operation::Operator)
    op = basis(state)==basis(operation) ? operation : embed(basis(state), indices, operation)
    state.data = (op*state*op').data
    state
end

function apply!(state::Ket, indices, operation::T) where {T<:AbstractSuperOperator}
    apply!(dm(state), indices, operation)
end

function apply!(state::Operator, indices, operation::T) where {T<:AbstractSuperOperator}
    op = basis(state)==basis(operation) ? operation : embed(basis(state), indices, operation)
    state.data = (op*state).data
    state
end
