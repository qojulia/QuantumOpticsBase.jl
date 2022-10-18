# LazyKet only used for numeric conversion of product states in QuantumCumulants.jl
import Base: isequal, == #, *, /, +, -

"""
    LazyKet(b, kets)

Lazy implementation of a tensor product of kets.

The subkets are stored in the `kets` field.
It is only used for the numeric conversion of product states in QuantumCumulants.jl.
"""
mutable struct LazyKet{B,T} <: StateVector{B,T}
    basis::B
    kets::T
    function LazyKet(b::B, kets::T) where {B<:CompositeBasis,T<:Tuple}
        N = length(b.bases)
        for n=1:N
            @assert isa(kets[n], StateVector)
            @assert kets[n].basis == b.bases[n] #
        end
        new{B,T}(b, kets)
    end
end
function LazyKet(b::CompositeBasis, kets::Vector)
    Base.depwarn("LazyKet(b, kets::Vector) is deprecated, use LazyKet(b, Tuple(kets)) instead.",
                :LazyKet; force=true)
    return LazyKet(b,Tuple(kets))
end

function expect(op::LazyTensor, state::LazyKet)
    ops = op.operators
    kets = state.kets
    @assert length(ops) == length(kets) && length(kets) > 1
    prod(expect(ops[i],kets[i]) for i=1:length(kets))
end

Base.copy(x::LazyKet) = LazyKet(x.basis, Tuple(copy(op) for op in x.kets))
isequal(x::LazyKet, y::LazyKet) = samebases(x,y) && isequal(x.kets, y.kets)
==(x::LazyKet, y::LazyKet) = samebases(x,y) && x.kets==y.kets
