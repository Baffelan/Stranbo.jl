# Stranbo

[![Build Status](https://github.com/gvdr/Stranbo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvdr/Stranbo.jl/actions/workflows/CI.yml?query=branch%3Amain)


This is a light package to create time series with anomalies. Its name is my dialect word for "strange".

The fundamental idea is to define stochastic process as (parametric) types, having a common interface to operate with them. Emphasis is on flexibility, composibility, and performance (we are in Julia, right?).

The difference with other existing packages is that we do not include in the package ANY analysis, fitting, or learning procedure. This is just about simulating (potentially complex) time series with given paramaters. This seems to be a functionality not exposed in any other package.

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
@kwdef struct Sarma{T<:Real} # T is the numeric type, say Float32, and it is mandatory
    p::Int = 1 # the auto-regressive order
    q::Int = 1 # the moving average order
    s::Int = 1 # the seasonality 
    ar::Vector{T} = [0.5] # a vector of coefficient for the ar process
    ma::Vector{T} = [0.5] # a vector of coefficient for the ma process
    dₙ = Normal(zero(T),one(T)) # the generative noise in the process, defaulting to a standard Normal
end
```

Each parameter can be changed, if needed, and otherwise be left to the default:
```Julia
Sarma(ar = [0.2], ma = [0.1])
```
define an arma (1,1) with ar and ma coefficients as specified (seasonality, p, and q are all set to the default 1).

One of the main functions in the package is `realise`, which takes an array of time-series components, eventually one or more additive noises, (and in future an observation function), and number of points to sample.

For example

```Julia
test_series = realise([
    Sarma{Float32}(ar = [.1], ma = [.1], dₙ = Normal(.0,.1)),
    Sarma{Float32}(ar = [.7], ma = [.7], s = 7, dₙ = Normal(.0,1.2)),
    mixed_dirac_normal(5.0,0.999)
    ], 
    L)
```

Define an sarima (1,0,1)x(1,0,1)7, with coefficients as defined and two level of noise with variance $0.1$ and a mixed dirac normal anomaly distribution with variance 5.0 and density $1-0.999$.

## State Space

The tooling before, together with base Julia functions, allows us to use the (s)ar(i)ma time series to build **simple** state space process with additive anomalies.
In the following we'll create a state space of the form

$$
\left\{\begin{matrix}
y_t = & G x_t + & \alpha_t \\ 
x_t = & T x_{t-1} + & \omega_t
\end{matrix}\right. \, .
$$

In our case, $x_t$ will be a 3-dimensional process, where each dimension follows an independent sarima process. The matrix $G$ is defined below, and $\alpha_t$ is an additive normal noise with density 0.001.

As above, define the lenght of the process and the anomaly noise distribution.
We realise a noise path in advance. In this way we can keep track of it in following analysis.
```Julia
L = 1000

αₜ = realise(
        mixed_dirac_normal(0.4,0.99),
        L
    )
```

Then, we define the observation matrix $G$.
We keep it simple, for the time being:

```Julia
G  = [0.2 0.3 0.5]
```

Each component of the $x_t$ is an independent sarma process.
We first define them as objects of Sarma type (I chose random parameter)

```Julia
proc1 = [
    Sarma{Float32}(ar = [.1], ma = [.1]),
    Sarma{Float32}(ar = [.3], ma = [.7], s = 7),
    Sarma{Float32}(ar = [.7], ma = [.3], s = 14, dₙ = Normal(.0,0.8))
    ]

proc2 = [
    Sarma{Float32}(ar = [.1], ma = [.1], dₙ = Normal(.0,.2)),
    Sarma{Float32}(ar = [.7], ma = [.3], s = 30, dₙ = Normal(.0,1.2))
    ]

proc3 = [
    Sarma{Float32}(p = 2, q = 2, ar = [.1,.02], ma = [.1,.02], dₙ = Normal(.0,.2)),
    Sarma{Float32}(p = 4, q = 4, ar = [.9,.001,.001,0.001], ma = [.8,.001,.001,0.001], s = 365)
    ]
```
Then, we realise $x_t$ process

```Julia
xₜ  = realise.([ # notice the broadcasting . (dot) following the function call!
        proc1,
        proc2,
        proc3
    ], 
    L)
```

Finally, we apply the observation matrix $G$ to the process and add the noise.

```Julia
yₜ = stack(test_series) * G' .+ αₜ
```

# TODO

- [ ] Non-linear observation processes
- [ ] Passing pre-computed noise (i.e., with MultiNormal)
- [ ] Multiplicative noises (and other non linear noises)
- [ ] Noises as stochastic processes
