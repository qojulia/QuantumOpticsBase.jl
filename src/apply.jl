function apply!(state::Ket, indices, operation::Operator)
    op = basis(state)==basis(operation) ? operation : embed(basis(state), indices, operation)
    if (length(indices)>1)
        for i in 2:length(indices)
            if indices[i]<indices[i-1]
                op = permutesystems(op, indices)
                break
            end
        end
    end
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
