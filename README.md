# Stranbo

[![Build Status](https://github.com/gvdr/Stranbo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvdr/Stranbo.jl/actions/workflows/CI.yml?query=branch%3Amain)


This is a light package to create time series with anomalies. Its name is my dialect word for "strange".

# Usage

## Installation

You can install the package from GitHub by typing:

```Julia
using Pkg
Pkg.add("")
```

## Create a series

For the moment, the package supports ar(i)ma processes with seasonalities and additive noise.

We first define the number of observations in the series:

```Julia
L = 10_000
```

Then, we define a distribution for the noise. For this, we can use whatever distribution we can define in `Distributions.jl`. `Stranbo` has defined an ad hoc mixed dirac+normal distribution:

```Julia
a = mixed_dirac_normal(0.4,0.99)
```

`a` is equal to $0$ with probability $0.99$, and is sampled from a normal distribution with mean 0 and variance $0.4$ with probability $1-0.99$.

For the (s)ar(i)ma process, its components are defined through objects of the `Stranbo` defined type `Sarma`.
These have the following structure:
```Julia
struct Sarma{T<:Real} # T is the numeric type, say Float32, and it is mandatory
    p::Int = 1 # the auto-regressive order
    q::Int = 1 # the moving average order
    s::Int = 1 # the seasonality 
    ar::Vector{T} = [0.5] # a vector of coefficient for the ar process
    ma::Vector{T} = [0.5] # a vector of coefficient for the ma process
    dₙ = Normal(zero(T),one(T)) # the generative noise in the process, defaulting to a standard Normal
end
```

The main function in the package is `realise`, which takes an array of time-series components, eventually one or more additive noises, (and in future an observation function), and number of points to sample.

For example

```Julia
test_series = realise([
    Sarma{Float32}(ar = [.1], ma = [.1], dₙ = Normal(.0,.1)),
    Sarma{Float32}(ar = [.7], ma = [.7], s = 7, dₙ = Normal(.0,1.2)),
    mixed_dirac_normal(5.0,0.999)
    ], 
    L)
```

Define an sarima (1,0,1)(1,0,1)7, with coefficients as defined and two level of noise with variance $0.1$ and a mixed dirac normal anomaly distribution with variance 5.0 and density 0.999.

# State Space

```Julia
L = 1000

a = mixed_dirac_normal(0.4,0.99)

test_series = realise.([
    [Sarma{Float32}(ar = [.1], ma = [.1], dₙ = Normal(.0,.1)),
    Sarma{Float32}(ar = [.7], ma = [.7], s = 7, dₙ = Normal(.0,1.2))],
    [Sarma{Float32}(ar = [.1], ma = [.1], dₙ = Normal(.0,.1)),
    Sarma{Float32}(ar = [.7], ma = [.7], s = 7, dₙ = Normal(.0,1.2))],
    [Sarma{Float32}(ar = [.1], ma = [.1], dₙ = Normal(.0,.1)),
    Sarma{Float32}(ar = [.7], ma = [.7], s = 7, dₙ = Normal(.0,1.2))]
    ], 
    L)

G  = [0.2 0.3 0.5]
xₜ = stack(test_series) * G'
aₜ = realise(mixed_dirac_normal(0.4,0.5),1000)
yₜ = xₜ .+ aₜ
```

# TODO

- [ ] Non-linear observation processes
- [ ] Multiplicative noises (and other non linear noises)
- [ ] Noises as stochastic processes
