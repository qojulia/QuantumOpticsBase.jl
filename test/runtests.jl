using TestItemRunner
using QuantumOpticsBase

testfilter = ti->(contains(ti.filename, "test_") && endswith(ti.filename, ".jl") )
@run_package_tests filter=testfilter
