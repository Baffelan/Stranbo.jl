module Stranbo

# Write your package code here.
using Bumper
using LinearAlgebra
using Random
using Distributions
using Polynomials
using StaticArrays
using StrideArrays


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
export laggedvector
export coeffpoly
export Δ, Uno
include("utils.jl")
    
end
