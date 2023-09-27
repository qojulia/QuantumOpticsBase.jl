function is_apply_shortcircuit(state, indices, operation)
    if nsubsystems(state) == 1
        basis(state)==basis(operation) || throw(ArgumentError("`apply!` failed due to incompatible bases of the state and the operation attempted to be applied on it"))
    end
    basis(state)==basis(operation) || return false
    j = 1
    for i in indices
        i == j || return false
        j+=1
    end
    return j-1 == length(indices)
end

function apply!(state::Ket, indices, operation::Operator)
    op = is_apply_shortcircuit(state, indices, operation) ? operation : embed(basis(state), indices, operation)
    state.data = (op*state).data
    state
end

function apply!(state::Operator, indices, operation::Operator)
    op = is_apply_shortcircuit(state, indices, operation) ? operation : embed(basis(state), indices, operation)
    state.data = (op*state*op').data
    state
end

function apply!(state::Ket, indices, operation::T) where {T<:AbstractSuperOperator}
    apply!(dm(state), indices, operation)
end

function apply!(state::Operator, indices, operation::T) where {T<:AbstractSuperOperator}
    if is_apply_shortcircuit(state, indices, operation)
        state.data = (operation*state).data
        return state
    else
        error("`apply!` does not yet support embedding superoperators acting only on a subsystem of the given state")
    end
end
