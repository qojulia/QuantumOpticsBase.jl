@testitem "charge" begin
    using QuantumOpticsBase

    @test_throws DimensionMismatch ChargeBasis(-1)

    @test_throws DimensionMismatch ShiftedChargeBasis(2, 1)

    ncut = rand(1:10)
    cb1 = ChargeBasis(ncut)
    cb2 = ChargeBasis(ncut)
    @test cb1 == cb2

    n1 = rand(-10:-1)
    n2 = rand(1:10)
    cb1 = ShiftedChargeBasis(n1, n2)
    cb2 = ShiftedChargeBasis(n1, n2)
    @test cb1 == cb2

    for cb in [ChargeBasis(5), ShiftedChargeBasis(-4, 5)]
        ψ = chargestate(cb, -3)
        eiφ = expiφ(cb)
        @test eiφ * ψ == chargestate(cb, -2)
        @test eiφ' * ψ == chargestate(cb, -4)
        @test expiφ(cb, k=3) * ψ == chargestate(cb, 0)
        @test expiφ(cb, k=9) * ψ == 0 * ψ
        @test expiφ(cb, k=-3) * ψ == 0 * ψ
        @test expiφ(cb, k=3) == eiφ^3
        @test expiφ(cb, k=11) == 0 * eiφ

        @test eltype(chargeop(Float64, cb).data) == Float64
        @test eltype(chargeop(ComplexF64, cb).data) == ComplexF64
        @test eltype(expiφ(Float64, cb).data) == Float64
        @test eltype(expiφ(ComplexF64, cb).data) == ComplexF64

        n = chargeop(cb)
        @test n * ψ == -3 * ψ
        @test n * eiφ - eiφ * n == eiφ

        ei3φ = expiφ(cb, k=3)
        @test cosφ(cb, k=3) == (ei3φ + ei3φ') / 2
        @test sinφ(cb, k=3) == (ei3φ - ei3φ') / 2im
    end

end  # testset
