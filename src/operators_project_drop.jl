# Project and drop functions for quantum measurements and post-selection

"""
    _drop_singular_bases(state)

Remove singular (dimension-1) bases from a quantum state or operator.
This is useful after measurement or post-selection operations that reduce
certain subsystems to classical states.

# Examples
```julia
# After measuring a subsystem in a definite state
reduced_ket = _drop_singular_bases(post_measurement_ket)
reduced_op = _drop_singular_bases(post_measurement_operator)
```
"""
function _drop_singular_bases(ket::Ket)
    b = tensor([b for b in basis(ket).bases if length(b)>1]...)
    return Ket(b, ket.data)
end

function _drop_singular_bases(op::Operator)
    b = tensor([b for b in basis(op).bases if length(b)>1]...)
    return Operator(b, op.data)
end

"""
    _branch_prob(state)

Calculate the probability of a quantum branch/measurement outcome.
Returns the square of the norm for kets, or the trace for density operators.
"""
_branch_prob(psi::Ket) = norm(psi)^2
_branch_prob(op::Operator) = real(sum((op.data[i, i] for i in 1:size(op.data,1))))

"""
    _overlap(left, right)

Calculate the overlap between two quantum states/operators.
For kets: |⟨left|right⟩|²
For ket-operator pairs: ⟨left|right|left⟩
"""
_overlap(l::Ket, r::Ket) = abs2(l'*r)
_overlap(l::Ket, op::Operator) = real(l'*op*l)

"""
    _project_and_drop(state, project_on, basis_index)

Project a quantum state onto a specific state in one subsystem and remove
that subsystem from the composite state. This is useful for conditional
evolution after measurement.

# Arguments
- `state`: Quantum state (Ket or Operator) to project
- `project_on`: State to project onto
- `basis_index`: Index of the subsystem to project and remove

# Returns
Reduced state with the projected subsystem removed.
"""
function _project_and_drop(state::Ket, project_on, basis_index)
    singularbasis = GenericBasis(1)
    singularket = basisstate(singularbasis,1)
    proj = projector(singularket, project_on')
    basis_r = collect(Any,basis(state).bases)
    basis_l = copy(basis_r)
    basis_l[basis_index] = singularbasis
    emproj = embed(tensor(basis_l...),tensor(basis_r...),basis_index,proj)
    result = emproj*state
    return _drop_singular_bases(result)
end

function _project_and_drop(state::Operator, project_on, basis_index)
    singularbasis = GenericBasis(1)
    singularket = basisstate(singularbasis,1)
    proj = projector(singularket, project_on')
    basis_r = collect(Any,basis(state).bases)
    basis_l = copy(basis_r)
    basis_l[basis_index] = singularbasis
    emproj = embed(tensor(basis_l...),tensor(basis_r...),basis_index,proj)
    result = emproj*state*emproj'
    return _drop_singular_bases(result)
end