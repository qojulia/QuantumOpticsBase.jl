using Test
using QuantumOpticsBase
using Random

@testset "state_definitions" begin

n=30
b=FockBasis(n)
omega=40.3
T=2.3756
r=thermalstate(omega*number(b),T)
for k=1:n-1
    @test isapprox(r.data[k+1,k+1]/r.data[k,k],exp(-omega/T))
end
S=entropy_vn(r)
z=sum(exp.(-[0:n;]*omega))
s=expect(omega*number(b),r)/T+log(z)
isapprox(S,s)

alpha=rand()+im*rand()
r=coherentthermalstate(b,omega*number(b),T,alpha)
@test isapprox(expect(number(b),r),abs(alpha)^2+1/(exp(omega/T)-1), atol=1e-14)
@test isapprox(expect(destroy(b),r),alpha, atol=1e-14)
@test isapprox(entropy_vn(r),S, atol=1e-14)

rp=phase_average(r)
@test isapprox(expect(number(b),rp),abs(alpha)^2+1/(exp(omega/T)-1), atol=1e-14)
@test isapprox(expect(destroy(b),rp),0, atol=1e-14)
for k=1:n
    @test isapprox(rp.data[k,k],r.data[k,k], atol=1e-14)
end

rpas=passive_state(r)
for k=1:n-1
    @test real(rpas.data[k+1,k+1])<real(rpas.data[k,k])
end
@test isapprox(expect(number(b),rpas),1/(exp(omega/T)-1), atol=1e-14)
@test isapprox(expect(destroy(b),rpas),0, atol=1e-14)
@test isapprox(entropy_vn(rpas),S, atol=1e-14)

Random.seed!(0)

for n in 1:5:50
    b = FockBasis(n)
    @test isapprox(norm(randstate_haar(b)), 1, atol=1e-8)
    U = randunitary_haar(b)
    @test isapprox(dagger(U)*U, identityoperator(b), atol=1e-8)
end

end # testset
