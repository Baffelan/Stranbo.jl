module Stranbo

import StatsBase: sample

using Bumper: @no_escape, alloc, default_buffer
using Distributions: Distribution, MixtureModel, Normal
using LinearAlgebra: ⋅
using Polynomials: *, -, Polynomial, coeffs
using Random: rand
using StaticArrays: SVector

@kwdef struct SARIMA{T<:Real}
    s::Int
    d::Int
    ar::SVN where {SVN <: SVector{N,T} where N}
    ma::SVM where {SVM <: SVector{M,T} where M}
    dₙ
end

export realise
export realise_all
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
