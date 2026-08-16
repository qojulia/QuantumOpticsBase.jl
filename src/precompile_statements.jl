# Generated from the cold-start scenarios on Julia 1.12.6 (x86_64-linux-gnu).
# Each raw trace was filtered with sortprecompile.py revision
# 4ea3c813d9a5adc494d84368e08825f5d765a0a6 using `-m 50.000001`.
# Timing comments preserve the largest observation for each exact statement.
# Recompilations and signatures tied to scenarios, value-specific FFT plans,
# generated names, compiler internals, or modules not bound in QuantumOpticsBase
# are excluded. These observations select statements; they are not benchmarks.

using PrecompileTools: @setup_workload

@setup_workload begin
    #=  106.9 ms =# precompile(Tuple{typeof(Base.:(*)), Array{Base.Complex{Float64}, 1}, LinearAlgebra.Adjoint{Base.Complex{Float64}, Array{Base.Complex{Float64}, 1}}})
    #=  105.5 ms =# precompile(Tuple{typeof(Base.:(*)), LinearAlgebra.Adjoint{Base.Complex{Float64}, SparseArrays.SparseMatrixCSC{Base.Complex{Float64}, Int64}}, Array{Base.Complex{Float64}, 2}, SparseArrays.SparseMatrixCSC{Base.Complex{Float64}, Int64}})
    #=   73.7 ms =# precompile(Tuple{typeof(Base.Broadcast.materialize), Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(Base.real), Tuple{Array{Base.Complex{Float64}, 2}}}})
    #=   67.3 ms =# precompile(Tuple{typeof(Base.:(/)), Array{Base.Complex{Float64}, 2}, Float64})
    #=   58.6 ms =# precompile(Tuple{typeof(Base.:(/)), Array{Base.Complex{Float64}, 2}, Int64})
end
