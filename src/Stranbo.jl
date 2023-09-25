module Stranbo

# Write your package code here.
using Random
using Distributions

export realise
include("glue.jl")

export mixed_dirac_normal
include("additive_anomaly.jl")

export Sarma
include("s_arma.jl")
    
end
