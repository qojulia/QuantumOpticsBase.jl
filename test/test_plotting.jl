@testitem "Bloch Sphere Plotting" tags=[:plotting] begin
    using QuantumOpticsBase
    using CairoMakie
 
    b = SpinBasis(1//2)
 

    function bloch_vector(ψ::Ket)
        α, β = ψ.data
        x = 2 * real(conj(α) * β)
        y = 2 * imag(conj(α) * β)
        z = abs2(α) - abs2(β)
        return (x, y, z)
    end
 

    @testset "blochsphere returns a Figure" begin
        @test blochsphere(spinup(b)) isa Figure
    end
 
    @testset "blochsphereplot! returns a Plot" begin
        fig = Figure()
        ax  = Axis3(fig[1, 1])
        @test blochsphereplot!(ax, spinup(b)) isa Makie.Plot
    end
 

    @testset "Bloch vector |0⟩ north pole" begin
        x, y, z = bloch_vector(spinup(b))
        @test x ≈  0 atol=1e-10
        @test y ≈  0 atol=1e-10
        @test z ≈  1 atol=1e-10
    end
 
    @testset "Bloch vector |1⟩ south pole" begin
        x, y, z = bloch_vector(spindown(b))
        @test x ≈  0 atol=1e-10
        @test y ≈  0 atol=1e-10
        @test z ≈ -1 atol=1e-10
    end
 
    @testset "Bloch vector |+⟩ +x pole" begin
        x, y, z = bloch_vector(normalize(spinup(b) + spindown(b)))
        @test x ≈  1 atol=1e-10
        @test y ≈  0 atol=1e-10
        @test z ≈  0 atol=1e-10
    end
 
    @testset "Bloch vector |-⟩ -x pole" begin
        x, y, z = bloch_vector(normalize(spinup(b) - spindown(b)))
        @test x ≈ -1 atol=1e-10
        @test y ≈  0 atol=1e-10
        @test z ≈  0 atol=1e-10
    end
 
    @testset "Bloch vector |+i⟩ +y pole" begin
        x, y, z = bloch_vector(normalize(spinup(b) + im*spindown(b)))
        @test x ≈  0 atol=1e-10
        @test y ≈  1 atol=1e-10
        @test z ≈  0 atol=1e-10
    end
 
    @testset "Bloch vector |-i⟩ -y pole" begin
        x, y, z = bloch_vector(normalize(spinup(b) - im*spindown(b)))
        @test x ≈  0 atol=1e-10
        @test y ≈ -1 atol=1e-10
        @test z ≈  0 atol=1e-10
    end
 

    @testset "Bloch vector matches θ/φ parameterisation" begin
        for (θ, φ) in [(π/3, π/4), (π/2, π/3), (2π/3, 3π/4), (π/4, 7π/6)]
            ψ = cos(θ/2)*spinup(b) + exp(im*φ)*sin(θ/2)*spindown(b)
            x, y, z = bloch_vector(ψ)
            @test x ≈ sin(θ)*cos(φ) atol=1e-10
            @test y ≈ sin(θ)*sin(φ) atol=1e-10
            @test z ≈ cos(θ)        atol=1e-10
        end
    end
 
    @testset "Bloch vector has unit length for any pure state" begin
        for (θ, φ) in [(0.0, 0.0), (π/4, π/3), (π/2, 0.0), (π/2, π), (π, 0.0)]
            ψ = cos(θ/2)*spinup(b) + exp(im*φ)*sin(θ/2)*spindown(b)
            x, y, z = bloch_vector(ψ)
            @test x^2 + y^2 + z^2 ≈ 1 atol=1e-10
        end
    end
 

    @testset "|0⟩ renders without error" begin
        fig = blochsphere(spinup(b))
        save("test_bloch_0.png", fig)
        @test isfile("test_bloch_0.png")
        rm("test_bloch_0.png")
    end
 
    @testset "|1⟩ renders without error" begin
        fig = blochsphere(spindown(b))
        save("test_bloch_1.png", fig)
        @test isfile("test_bloch_1.png")
        rm("test_bloch_1.png")
    end
 
    @testset "|+⟩ renders without error" begin
        fig = blochsphere(normalize(spinup(b) + spindown(b)))
        save("test_bloch_plus.png", fig)
        @test isfile("test_bloch_plus.png")
        rm("test_bloch_plus.png")
    end
 
    @testset "|-⟩ renders without error" begin
        fig = blochsphere(normalize(spinup(b) - spindown(b)))
        save("test_bloch_minus.png", fig)
        @test isfile("test_bloch_minus.png")
        rm("test_bloch_minus.png")
    end
 
    @testset "|+i⟩ renders without error" begin
        fig = blochsphere(normalize(spinup(b) + im*spindown(b)))
        save("test_bloch_plusi.png", fig)
        @test isfile("test_bloch_plusi.png")
        rm("test_bloch_plusi.png")
    end
 
    @testset "|-i⟩ renders without error" begin
        fig = blochsphere(normalize(spinup(b) - im*spindown(b)))
        save("test_bloch_minusi.png", fig)
        @test isfile("test_bloch_minusi.png")
        rm("test_bloch_minusi.png")
    end
 

    @testset "Arbitrary state θ=π/3 φ=π/4 renders without error" begin
        ψ = cos(π/6)*spinup(b) + exp(im*π/4)*sin(π/6)*spindown(b)
        fig = blochsphere(ψ)
        save("test_bloch_arb1.png", fig)
        @test isfile("test_bloch_arb1.png")
        rm("test_bloch_arb1.png")
    end
 
    @testset "Arbitrary state θ=π/2 φ=π/3 renders without error" begin
        ψ = cos(π/4)*spinup(b) + exp(im*π/3)*sin(π/4)*spindown(b)
        fig = blochsphere(ψ)
        save("test_bloch_arb2.png", fig)
        @test isfile("test_bloch_arb2.png")
        rm("test_bloch_arb2.png")
    end
 

    @testset "Custom arrowcolor and spherealpha" begin
        fig = blochsphere(spinup(b); arrowcolor=:blue, spherealpha=0.3)
        save("test_bloch_custom_color.png", fig)
        @test isfile("test_bloch_custom_color.png")
        rm("test_bloch_custom_color.png")
    end
 
    @testset "Custom spherecolor" begin
        fig = blochsphere(spindown(b); spherecolor=:pink, spherealpha=0.2)
        save("test_bloch_custom_sphere.png", fig)
        @test isfile("test_bloch_custom_sphere.png")
        rm("test_bloch_custom_sphere.png")
    end
 
    @testset "Wireframe, labels and axes toggled off" begin
        fig = blochsphere(spinup(b); showwireframe=false, showlabels=false, showaxes=false)
        save("test_bloch_minimal.png", fig)
        @test isfile("test_bloch_minimal.png")
        rm("test_bloch_minimal.png")
    end
 

    @testset "Wrong dimension state throws error" begin
        b3  = SpinBasis(1)
        ψ_3 = basisstate(b3, 1)
        @test_throws ErrorException blochsphere(ψ_3)
    end
end