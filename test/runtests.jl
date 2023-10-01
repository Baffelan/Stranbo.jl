using Stranbo
using Test
using Polynomials
using StaticArrays
using Distributions

@testset "Stranbo.jl" begin
    @test_throws UndefKeywordError SARIMA()
    @test typeof(SARIMA{Float64}(1, 0, SVector{0,Float64}(), SVector{0,Float64}(), Normal{Float64}(0.0, 1.0)))  <: SARIMA
end

@testset "s_arma.jl" begin
    @test typeof(sarima()) <: SARIMA
end

@testset "utils.jl" begin
    @test Stranbo.One == Polynomial([1],:B)
end

# @testset "glue.jl" begin
#     # Write your tests here.
# end

# @testset "additive_anomaly.jl" begin
#     # Write your tests here.
# end
