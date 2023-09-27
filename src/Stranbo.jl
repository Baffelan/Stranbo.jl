module Stranbo

# Write your package code here.
using Random
using Distributions
using Polynomials

export realise
include("glue.jl")

export mixed_dirac_normal
include("additive_anomaly.jl")

export Sarma
include("s_arma.jl")

export getidx
export lag
export pushedback_sum
export laggedvector
export coeffpoly
export Δ, Uno
include("utils.jl")
    
end
