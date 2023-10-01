# Stranbo

[![Build Status](https://github.com/Baffelan/Stranbo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Baffelan/Stranbo.jl/actions/workflows/CI.yml?query=branch%3Amain)


The aim of this nimble package is to simulate time series with anomalies. Its named after the word for "strange" in my dialect.

The fundamental idea is to define stochastic process as (parametric) types, sharing a common interface to operate with them. Emphasis is on flexibility, composibility, and performance (we are in Julia, right?).

The main difference with other existing packages is that we do not include in the package ANY analysis, fitting, or learning procedure. This is just about simulating (potentially complex) time series with given paramaters and anomaly regimes. This seems to be a functionality not exposed in any other package (that we are aware of).

# Implemented processes

`Stranbo` comes with a bunch of pre-cooked stochastic processes, that can be mixed with other user-defined processes.

So far we have builder functions for:

- `sarima`: with this you can create any combination of seasonal, auto-regressive, integrated, moving average (generated from gaussian noise, whatever other noise you can define as a `Distribution`, or any deterministic vector of the right size). The process can be any composition of $(p,d,q)s$ components (so you can have as many seasonal components, integrated or not, that you want). 
- `sarimax`: as for `sarima` but with added auxiliary components (that also have $(p,d,q)s$ parameters. Also for `sarimax`, you can mix and match as you want.
- `mixed_dirac_normal`: that's there for generating anomalies. It's constantly zero most of the time, and sometimes samples from a normal distribution with given $\sigma$ and zero mean.



# Usage

# WARNING

The package is in very rapid development, the docs are being updated as we go.

## Installation

You can install the package from GitHub by typing:

```Julia
using Pkg
Pkg.add("https://github.com/Baffelan/Stranbo.jl")
```

## Create a series

For the moment, the package supports sarima(x) processes with seasonalities and additive noise.

We first define the number of observations in the series:

```Julia
L = 10_000
```

Then, we define a distribution for the noise. For this, we can use whatever distribution we can define in `Distributions.jl`. `Stranbo` has defined an ad hoc mixed dirac+normal distribution:

```Julia
a = mixed_dirac_normal(0.4,0.01)
```

`a` is equal to $0$ with probability $0.99$, and is sampled from a normal distribution with mean 0 and variance $0.4$ with probability $1-0.99$.

For the (s)ar(i)ma process, its components are defined through objects of the `Stranbo` defined type `SARIMA`.
These have the following structure:
```Julia
@kwdef struct SARIMA{T<:Real} # T defines the numerical precision
    s::Int # The seasonality effect
    d::Int # The integration order (can be 0)
    ar <: SVector{N,T} # The auto-regressive coefficients (given as a static vector)
    ma <: SVector{M,T} # The moving-average coefficients (given as a static vector)
    dₙ # the generating noise (as a distribution or a vector)
end
```

The orders $p$ and $q$ are inferred by the lenght of the $ar$ and $ma$ vectors.

We also have a convenience constructor `sarima` with some defaults. The `sarima()` interface offers an easy way to define the building blocks of a SARIMA process without needing to specify every detail.

```Julia
function  sarima(;
    T::Type{<:Real} = Float64,
    s::Int = 1,
    d::Int = 0,
    ar = T[],
    ma = T[],
    dₙ = Normal(zero(T),one(T)))

    SARIMA{T}(s,d,ar,ma,dₙ)

end
```

When calling `sarima()` each parameter can be changed, if needed, and otherwise be left to the default. For example:
```Julia
sarima(ar = [0.2], ma = [0.1])
```
define an arma (1,1) with ar and ma coefficients as specified (seasonality, p, and q are all set to the default 1).

And:
```Julia
sarima(d = 2, ar = [0.2], ma = [0.1])
```
define an arima (1,2,1) with ar and ma coefficients as specified.

To define a `sarimax` things are mostly the same, but you need to defin two more parameters: the auxiliary input and the related process coefficients.

One of the main functions in the package is `sample`, which takes an array of time-series components, eventually one or more additive noises, (and in future an observation function), and number of points to sample.

For example, we can define a sarima (1,1,1)x(0,1,1)7 as follows:

```Julia
exmpl_sarima = [
        sarima(d = 1, ar = [.1], ma = [.1]),
        sarima(d = 1, ma = [.7], s = 7)
    ]
```

and from that we can either samples of the sarima trajectory as:

```Julia
sample(exmpl_sarima, 100_000)
```

Or add to it an additive noise by using `realise_all()`:

```Julia
realise_all([
    exmpl_sarima, # this is a vector of sarimas components
    a],
    100_000)
```

Notice that all the sarimas component should be passed wrapped in an array (so to disambiguate about generative noise, i.e., $z$, and additive anomalies, i.e. $a$).

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
        mixed_dirac_normal(0.4,0.01),
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
    sarima(ar = [.1], ma = [.1]),
    sarima(ma = [.7], s = 7),
    sarima(ma = [.3], s = 14)
    ]

proc2 = [
    sarima(ar = [.1], ma = [.1], dₙ = Normal(.0,.2)),
    sarima(ar = [.2], ma = [.2], s = 30)
    ]

proc3 = [
    sarima(ar = [.1,.02], ma = [.1,.02], dₙ = Normal(.0,.2)),
    sarima(ar = [.9,.001,.001,0.001], ma = [.8,.001,.001,0.001], s = 365)
    ]
```
Then, we sample a trajectory from each of the $x_t$ process

```Julia
xₜ  = sample.([ # notice the broadcasting . (dot) following the function call!
        proc1,
        proc2,
        proc3
    ], 
    L)
```

This is done pretty quickly:

```Julia
 Range (min … max):  5.796 ms …   9.324 ms  ┊ GC (min … max):  0.00% … 15.85%
 Time  (median):     6.752 ms               ┊ GC (median):    11.33%
 Time  (mean ± σ):   6.452 ms ± 548.207 μs  ┊ GC (mean ± σ):   6.87% ±  5.87%

   █▂
  ▆██▆▄▄▃▂▂▂▂▁▁▂▁▁▂▁▂▂▁▁▁▁▁▁▁▁▂▃▄▄▄▆██▇▆▅▄▃▄▃▃▁▂▃▂▂▁▂▂▁▁▁▁▁▁▂ ▃
  5.8 ms          Histogram: frequency by time        7.58 ms <

 Memory estimate: 23.84 MiB, allocs estimate: 9318.
 ```

Finally, we apply the observation matrix $G$ to the process and add the noise.

```Julia
yₜ = stack(xₜ) * G' .+ αₜ
```

We could have used whatever else function on `stack(xₜ)`, for example any non-linear function.

## A sarimax example

We first define two different auxiliary inputs:

```Julia
up_and_downs = sign.(cos.(range(0,24π,length=100_000))
wavey = sin.(range(0,6π,length=100_000))
```

With those, we can define two components for a sarimax process:

```Julia
proc_1 = sarimax(ar = [.8], ma = [2.], ax = [.5], dₙ = Normal(0.,0.2), axₙ = up_and_downs)
proc_1 = sarimax(ar = [.8], ma = [2.], ax = [.5], s = 200, dₙ = Normal(0.,0.2), axₙ = up_and_downs)
```

And we plot them:

```Julia
sample([test,testb],100_000) |> plot
```

# TODO

- [ ] switchin-markov models
- [ ] Multiplicative noises (and other non linear noises)
- [ ] Noises as stochastic processes
