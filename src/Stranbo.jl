module Stranbo

# Write your package code here.
using Bumper: alloc, default_buffer, @no_escape
using LinearAlgebra: ⋅
using Random: rand
using Distributions: Distribution, MixtureModel, Normal
using Polynomials: Polynomial, *, -, coeffs
using StaticArrays: SVector

export realise
include("glue.jl")

export mixed_dirac_normal
include("additive_anomaly.jl")

export SARIMA, sarima
export sample
include("s_arma.jl")

export getidx
export lag
export pushedback_sum
export seasonal_vector
export coeffpoly
export Δ
include("utils.jl")
    
end
