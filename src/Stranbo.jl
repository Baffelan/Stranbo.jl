module Stranbo

import StatsBase: sample

using Bumper: @no_escape, alloc, default_buffer, AllocBuffer
using Distributions: Distribution, MixtureModel, Normal
using LinearAlgebra: ⋅
using Polynomials: *, -, Polynomial, coeffs
using Random: rand
using StaticArrays: SVector


export ArrayParams
struct ArrayParams
    vector::V where {V <: AbstractArray{<:Number}}
    coefficients::C where {C <: AbstractArray{<:Number}}
end

export mixed_dirac_normal
include("additive_anomaly.jl")

export SARIMA
export sarima
export sample
include("s_arma.jl")

export SARIMAX
export sarimax
include("s_armax.jl")

export getidx
export backwarded_sum
export seasonal_vector
export Δ
include("utils.jl")


export realise
export realise_all
include("glue.jl")

    
end
