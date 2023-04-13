import Base: isequal, ==, *, /, +, -

"""
    LazyProduct(operators[, factor=1])
    LazyProduct(op1, op2...)

Lazy evaluation of products of operators.

The factors of the product are stored in the `operators` field. Additionally a
complex factor is stored in the `factor` field which allows for fast
multiplication with numbers.
"""

mutable struct LazyProduct{BL,BR,F,T,KTL,BTR} <: LazyOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factor::F
    operators::T
    ket_l::KTL
    bra_r::BTR
    function LazyProduct{BL,BR,F,T,KTL,BTR}(operators::T, ket_l::KTL, bra_r::BTR, factor::F=1) where {BL,BR,F,T,KTL,BTR}
        for i = 2:length(operators)
            check_multiplicable(operators[i-1], operators[i])
        end
        new(operators[1].basis_l, operators[end].basis_r, factor, operators,ket_l,bra_r)
    end
end
function LazyProduct(operators::T, factor::F=1) where {T,F}
    BL = typeof(operators[1].basis_l)
    BR = typeof(operators[end].basis_r)
    ket_l=Tuple(Ket(operators[i].basis_l) for i in 2:length(operators))
    bra_r=Tuple(Bra(operators[i].basis_r) for i in 1:length(operators)-1)
    KTL = typeof(ket_l)
    BTR = typeof(bra_r)
    LazyProduct{BL,BR,F,T,KTL,BTR}(operators, ket_l, bra_r, factor)
end




LazyProduct(operators::Vector{T}, factor=1) where T<:AbstractOperator = LazyProduct((operators...,), factor)
LazyProduct(operators::AbstractOperator...) = LazyProduct((operators...,))
LazyProduct() = throw(ArgumentError("LazyProduct needs at least one operator!"))

Base.copy(x::T) where T<:LazyProduct = T(([copy(op) for op in x.operators]...,),x.ket_l,x.bra_r, x.factor)
Base.eltype(x::LazyProduct) = promote_type(eltype(x.factor), eltype.(x.operators)...)

dense(op::LazyProduct) = op.factor*prod(dense.(op.operators))
dense(op::LazyProduct{B1,B2,F,T}) where {B1,B2,F,T<:Tuple{AbstractOperator}} = op.factor*dense(op.operators[1])
SparseArrays.sparse(op::LazyProduct) = op.factor*prod(sparse.(op.operators))
SparseArrays.sparse(op::LazyProduct{B1,B2,F,T}) where {B1,B2,F,T<:Tuple{AbstractOperator}} = op.factor*sparse(op.operators[1])

isequal(x::LazyProduct{B1,B2}, y::LazyProduct{B1,B2}) where {B1,B2} = (samebases(x,y) && isequal(x.operators, y.operators) && isequal(x.factor, y.factor))
==(x::LazyProduct{B1,B2}, y::LazyProduct{B1,B2}) where {B1,B2} = (samebases(x,y) && x.operators==y.operators && x.factor == y.factor)

# Arithmetic operations
-(a::T) where T<:LazyProduct = T(a.operators,a.ket_l,a.bra_r, -a.factor)

*(a::LazyProduct{B1,B2}, b::LazyProduct{B2,B3}) where {B1,B2,B3} = LazyProduct((a.operators..., b.operators...), a.factor*b.factor)
*(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor*b)
*(a::Number, b::LazyProduct) = LazyProduct(b.operators, a*b.factor)

/(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor/b)


dagger(op::LazyProduct) = LazyProduct(dagger.(reverse(op.operators)), conj(op.factor))

tr(op::LazyProduct) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. dense() or sparse()."))

ptrace(op::LazyProduct, indices) = throw(ArgumentError("Partial trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. dense() or sparse()."))

permutesystems(op::LazyProduct, perm::Vector{Int}) = LazyProduct(([permutesystems(op_i, perm) for op_i in op.operators]...,), op.factor)

identityoperator(::Type{LazyProduct}, ::Type{S}, b1::Basis, b2::Basis) where S<:Number = LazyProduct(identityoperator(S, b1, b2))



function tensor(a::Operator{B1,B2},b::LazyProduct{B3, B4, F, T, KTL, BTR}) where {B1,B2,B3,B4, F, T, KTL, BTR}
    ops = ([(i == 1 ? a : identityoperator(a.basis_r) ) ⊗ op for (i,op) in enumerate(b.operators)]...,)
    LazyProduct(ops,b.factor)
end
function tensor(a::LazyProduct{B1, B2, F, T, KTL, BTR},b::Operator{B3,B4}) where {B1,B2,B3,B4, F, T, KTL, BTR}
    ops = ([op ⊗ (i == length(a.operators) ? b : identityoperator(a.basis_l) ) for (i,op) in enumerate(a.operators)]...,)
    LazyProduct(ops,a.factor)
end

function mul!(result::Ket{B1},a::LazyProduct{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2}
    if length(a.operators)==1
        mul!(result,a.operators[1],b,a.factor*alpha,beta)
    else
        mul!(a.ket_l[end],a.operators[end],b,a.factor,0)
        for i=length(a.operators)-2:-1:1
            mul!(a.ket_l[i],a.operators[i+1],a.ket_l[i+1])
        end
        mul!(result,a.operators[1],a.ket_l[1],alpha,beta)
    end
    return result
end

function mul!(result::Bra{B2},a::Bra{B1},b::LazyProduct{B1,B2},alpha,beta) where {B1,B2}
    if length(b.operators)==1
        mul!(result, a, b.operators[1],b.factor*alpha,beta)
    else
        mul!(b.bra_r[1], a, b.operators[1], b.factor,0)
        for i=2:length(b.operators)-1
            mul!(b.bra_r[i],b.bra_r[i-1],b.operators[i])
        end
        mul!(result,b.bra_r[end],b.operators[end],alpha,beta)
    end
    return result
end

function mul!(result::Operator{B1,B3,T},a::LazyProduct{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3,T}
    if length(a.operators) == 1
        mul!(result,a.operators[1],b,a.factor*alpha,beta)
    else
        tmp1 = Operator(a.operators[end].basis_l,b.basis_r,similar(result.data,length(a.operators[end].basis_l),length(b.basis_r)))
        mul!(tmp1,a.operators[end],b,a.factor,0)
        for i=length(a.operators)-1:-1:2
            tmp2 = Operator(a.operators[i].basis_l, b.basis_r, similar(result.data,length(a.operators[i].basis_l),length(b.basis_r)))
            mul!(tmp2,a.operators[i],tmp1)
            tmp1 = tmp2
        end
        mul!(result,a.operators[1],tmp1,alpha,beta)
    end
    return result
end

function mul!(result::Operator{B1,B3,T},a::Operator{B1,B2},b::LazyProduct{B2,B3},alpha,beta) where {B1,B2,B3,T}
    if length(b.operators) == 1
        mul!(result, a, b.operators[1],b.factor*alpha,beta)
    else
        tmp1 = Operator(a.basis_l,b.operators[1].basis_r,similar(result.data,length(a.basis_l),length(b.operators[1].basis_r)))
        mul!(tmp1,a,b.operators[1],b.factor,0)
        for i=2:length(b.operators)-1
            tmp2 = Operator(a.basis_l,b.operators[i].basis_r,similar(result.data,length(a.basis_l),length(b.operators[i].basis_r)))
            mul!(tmp2,tmp1,b.operators[i])
            tmp1 = tmp2
        end
        mul!(result,tmp1,b.operators[end],alpha,beta)
    end
    return result
end
