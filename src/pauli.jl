import Base: isapprox
import QuantumInterface: PauliBasis

"""
    Base class for Pauli transfer matrix classes.
"""
abstract type PauliTransferMatrix{B1, B2} end


"""
    DensePauliTransferMatrix(B1, B2, data)

DensePauliTransferMatrix stored as a dense matrix.
"""
mutable struct DensePauliTransferMatrix{B1,B2,T<:Matrix} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DensePauliTransferMatrix(basis_l::BL, basis_r::BR, data::T) where {BL,
                                                                                BR,
                                                                                T<:Matrix}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new{BL, BR, T}(basis_l, basis_r, data)
    end
end

PauliTransferMatrix(ptm::DensePauliTransferMatrix) = ptm

function *(ptm0::DensePauliTransferMatrix{B, B},ptm1::DensePauliTransferMatrix{B, B}) where B
    return DensePauliTransferMatrix(ptm0.basis_l, ptm1.basis_r, ptm0.data*ptm1.data)
end

"""
    Base class for χ (process) matrix classes.
"""
abstract type ChiMatrix{B1, B2} end

"""
    DenseChiMatrix(b, b, data)

DenseChiMatrix stored as a dense matrix.
"""
mutable struct DenseChiMatrix{B1,B2,T<:Matrix} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DenseChiMatrix(basis_l::BL, basis_r::BR, data::T) where {BL,BR,T<:Matrix}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new{BL, BR, T}(basis_l, basis_r, data)
    end
end

ChiMatrix(chi_matrix::DenseChiMatrix) = chi_matrix

"""
The phase of a one-qubit Pauli product, indexed by
`4left_digit + right_digit + 1`.
"""
const pauli_multiplication_phases = (
    1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im,
    1.0 + 0.0im, 1.0 + 0.0im, 0.0 + 1.0im, 0.0 - 1.0im,
    1.0 + 0.0im, 0.0 - 1.0im, 1.0 + 0.0im, 0.0 + 1.0im,
    1.0 + 0.0im, 0.0 + 1.0im, 0.0 - 1.0im, 1.0 + 0.0im,
)

const ASCII_ZERO = UInt8('0')
const ASCII_THREE = UInt8('3')

@inline function base4_digit(digit::UInt8)
    ASCII_ZERO <= digit <= ASCII_THREE ||
        throw(ArgumentError("Pauli strings must contain only base-4 digits from 0 to 3."))
    return Int(digit - ASCII_ZERO)
end

@inline function pauli_multiplication_phase(left::Int, right::Int, num_qubits::Int)::ComplexF64
    phase = one(ComplexF64)
    for _ in 1:num_qubits
        left, left_digit = divrem(left, 4)
        right, right_digit = divrem(right, 4)
        phase *= pauli_multiplication_phases[4left_digit + right_digit + 1]
    end
    return phase
end

"""
    multiply_pauli_matrices(i4::String, j4::String)

A function to algebraically determine result of multiplying two
(N-qubit) Pauli matrices. Each Pauli matrix is represented by a string
in base 4. For example, σ₃⊗σ₀⊗σ₂ would be "302". The product of any pair of
Pauli matrices will itself be a Pauli matrix multiplied by any of the 1/4 roots
of 1.
"""
function multiply_pauli_matrices(i4::String, j4::String)
    i4_digits = codeunits(i4)
    j4_digits = codeunits(j4)
    isempty(i4_digits) && throw(ArgumentError("Pauli strings must be nonempty."))
    length(i4_digits) == length(j4_digits) ||
        throw(ArgumentError("Pauli strings must have equal lengths."))

    product = Vector{UInt8}(undef, length(i4_digits))
    phase = one(ComplexF64)
    for index in eachindex(i4_digits, j4_digits)
        left_digit = base4_digit(i4_digits[index])
        right_digit = base4_digit(j4_digits[index])
        product[index] = ASCII_ZERO + xor(left_digit, right_digit)
        phase *= pauli_multiplication_phases[4left_digit + right_digit + 1]
    end
    return String(product), phase
end

const multiply_pauli_matirices = multiply_pauli_matrices

function *(chi_matrix0::DenseChiMatrix{B, B},chi_matrix1::DenseChiMatrix{B, B}) where B

    num_qubits = length(chi_matrix0.basis_l[1].shape)
    sop_dim = 4 ^ num_qubits
    ret = zeros(ComplexF64, (sop_dim, sop_dim))

    phase_lookup = Matrix{ComplexF64}(undef, sop_dim, sop_dim)
    for right in 0:(sop_dim-1), left in 0:(sop_dim-1)
        phase_lookup[left+1, right+1] = pauli_multiplication_phase(left, right, num_qubits)
    end

    for ijkl in Iterators.product(0:(sop_dim-1),
                                  0:(sop_dim-1),
                                  0:(sop_dim-1),
                                  0:(sop_dim-1))
        i, j, k, l = ijkl
        if (chi_matrix0.data[i+1, j+1] != 0.0) & (chi_matrix1.data[k+1, l+1] != 0.0)
            ret[xor(i, k)+1, xor(l, j)+1] += (phase_lookup[i+1, k+1] * phase_lookup[l+1, j+1] * chi_matrix0.data[i+1, j+1] * chi_matrix1.data[k+1, l+1])
        end
    end
    return DenseChiMatrix(chi_matrix0.basis_l, chi_matrix0.basis_r, ret / 2^num_qubits)
end


# TODO MAKE A GENERATOR FUNCTION
"""
    pauli_operators(num_qubits::Integer)

Generate a list of N-qubit Pauli operators.
"""
function pauli_operators(num_qubits::Integer)
    pauli_funcs = (identityoperator, sigmax, sigmay, sigmaz)
    po = []
    for paulis in Iterators.product((pauli_funcs for _ in 1:num_qubits)...)
        basis_vector = reduce(⊗, f(SpinBasis(1//2)) for f in paulis)
        push!(po, basis_vector)
    end
    return po
end

"""
    pauli_basis_vectors(num_qubits::Integer)

Generate a matrix of basis vectors in the Pauli representation given a number
of qubits.
"""
function pauli_basis_vectors(num_qubits::Integer)
    num_qubits > 0 || throw(ArgumentError("Number of qubits must be positive."))

    basis = SpinBasis(1//2)
    paulis = (
        identityoperator(SparseOpType, ComplexF64, basis).data,
        sigmax(basis).data,
        sigmay(basis).data,
        sigmaz(basis).data,
    )
    sop_dim = 4 ^ num_qubits
    columns = Vector{SparseVector{ComplexF64,Int}}(undef, sop_dim)
    for column in eachindex(columns)
        code = column - 1
        data = paulis[mod(code, 4) + 1]
        code = div(code, 4)
        for _ in 2:num_qubits
            data = kron(paulis[mod(code, 4) + 1], data)
            code = div(code, 4)
        end
        columns[column] = sparse(reshape(data, sop_dim))
    end
    return reduce(hcat, columns)
end

"""
    PauliTransferMatrix(sop::DenseSuperOpType)

Convert a superoperator to its representation as a Pauli transfer matrix.
"""
function PauliTransferMatrix(sop::DenseSuperOpType)
    num_qubits = length(sop.basis_l[1].bases)
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
    num_qubits = length(ptm.basis_l[1].bases)
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
    num_qubits = length(unitary.basis_l.bases)
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

"""Equality for all varieties of superoperators."""
==(sop1::T, sop2::T) where T<:Union{DensePauliTransferMatrix, DenseSuperOpType, DenseChiMatrix} = sop1.data == sop2.data
==(sop1::Union{DensePauliTransferMatrix, DenseSuperOpType, DenseChiMatrix}, sop2::Union{DensePauliTransferMatrix, DenseChiMatrix}) = false

"""Approximate equality for all varieties of superoperators."""
function isapprox(sop1::T, sop2::T; kwargs...) where T<:Union{DensePauliTransferMatrix, DenseChiMatrix}
    return isapprox(sop1.data, sop2.data; kwargs...)
end
