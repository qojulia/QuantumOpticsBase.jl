import Base: isapprox


function PauliTransferMatrix(sop::DenseSuperOpType)
    num_qubits = nsubsystems(sop.basis_l[1])
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 4 ^ num_qubits
    data = real.(pbv' * sop.data * pbv / √sop_dim)
    return DensePauliTransferMatrix(sop.basis_l, sop.basis_r, data)
end

SuperOperator(unitary::DenseOpType) = spre(unitary) * spost(unitary')
SuperOperator(sop::DenseSuperOpType) = sop

"""
    SuperOperator(ptm::DensePauliTransferMatrix)

Convert a Pauli transfer matrix to its representation as a superoperator.
"""
function SuperOperator(ptm::DensePauliTransferMatrix)
    num_qubits = nsubsystems(ptm.basis_l[1])
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 4 ^ num_qubits
    data = pbv * ptm.data * pbv' / √sop_dim
    return DenseSuperOperator(ptm.basis_l, ptm.basis_r, data)
end

"""
    PauliTransferMatrix(unitary::DenseOpType)

Convert an operator, presumably a unitary operator, to its representation as a
Pauli transfer matrix.
"""
PauliTransferMatrix(unitary::DenseOpType) = PauliTransferMatrix(SuperOperator(unitary))

"""
    ChiMatrix(unitary::DenseOpType)

Convert an operator, presumably a unitary operator, to its representation as a χ matrix.
"""
function ChiMatrix(unitary::DenseOpType)
    num_qubits = nsubsystems(unitary.basis_l)
    pbv = pauli_basis_vectors(num_qubits)
    aj = pbv' * reshape(unitary.data, 4 ^ num_qubits)
    return DenseChiMatrix((unitary.basis_l, unitary.basis_l), (unitary.basis_r, unitary.basis_r), aj * aj' / (2 ^ num_qubits))
end

"""
    ChiMatrix(sop::DenseSuperOpType)

Convert a superoperator to its representation as a Chi matrix.
"""
function ChiMatrix(sop::DenseSuperOpType{B, B, T}) where {B, T}
    num_qubits = length(sop.basis_l)
    sop_dim = 4 ^ num_qubits
    po = pauli_operators(num_qubits)
    data = Matrix{eltype(T)}(undef, (sop_dim, sop_dim))
    for (idx, jdx) in Iterators.product(1:sop_dim, 1:sop_dim)
        data[idx, jdx] = tr((spre(po[idx]) * spost(po[jdx])).data' * sop.data) / √sop_dim
    end
    return DenseChiMatrix(sop.basis_l, sop.basis_r, data)
end

"""
    PauliTransferMatrix(chi_matrix::DenseChiMatrix)

Convert a χ matrix to its representation as a Pauli transfer matrix.
"""
function PauliTransferMatrix(chi_matrix::DenseChiMatrix{B, B, T}) where {B, T}
    num_qubits = length(chi_matrix.basis_l)
    sop_dim = 4 ^ num_qubits
    po = pauli_operators(num_qubits)
    data = Matrix{real(eltype(T))}(undef, (sop_dim, sop_dim))
    for (idx, jdx) in Iterators.product(1:sop_dim, 1:sop_dim)
        data[idx, jdx] = tr(mapreduce(x -> po[idx] * po[x[1]] * po[jdx] * po[x[2]] * chi_matrix.data[x[1], x[2]],
                                      +,
                                      Iterators.product(1:16, 1:16)).data) / sop_dim |> real
    end
    return DensePauliTransferMatrix(chi_matrix.basis_l, chi_matrix.basis_r, data)
end

"""
    SuperOperator(chi_matrix::DenseChiMatrix)

Convert a χ matrix to its representation as a superoperator.
"""
SuperOperator(chi_matrix::DenseChiMatrix) = SuperOperator(PauliTransferMatrix(chi_matrix))

"""
    ChiMatrix(ptm::DensePauliTransferMatrix)

Convert a Pauli transfer matrix to its representation as a χ matrix.
"""
ChiMatrix(ptm::DensePauliTransferMatrix) = ChiMatrix(SuperOperator(ptm))
