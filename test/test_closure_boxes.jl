@testitem "test_closure_boxes" begin
using QuantumOpticsBase
using Test

if VERSION >= v"1.14"
    @test isempty(Test.detect_closure_boxes(QuantumOpticsBase))
end
end
