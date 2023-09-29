# this function allows to index into a vector with zero or negative indices
# we use it when we build x[t] as a sum of elements x[t-k]
# and it allow us not to have to check whether k >= t
function getidx(v::V,i::Int) where V<:Vector{T} where T<:Number
    vᵢ = i > 0 ? view(v,i) : zero(T)
    return vᵢ
end

# extension of basic getidx to array of indices
getidx(v,Idx::I) where I<:AbstractArray  = [getidx(v,i) for i in Idx]
getidx(v,Idx::I) where I<:Base.OneTo{<:Number}  = [getidx(v,i) for i in Idx]

# given a vector `x`, an index `t` in `x` (the *current time*), and a vector of indexes `I`
# returns the backshifted view `x[t-i]` for `i ∈ l`
backshifted_view(x,t,I) = getidx(x,t .- I)

# the constant polynomial $1$ in the variable `:B`
const One = Polynomial([1],:B)

# we define the Lag operator polynomial, as $(1-B^s)^d$
# this is the special case for `s = 1`
Δ(d) = Polynomial(Int[1,-1],:B)^d

# we define the Lag operator polynomial, as $(1-B^s)^d$
Δ(; s = 1,d) = Polynomial(vcat(1,zeros(Int,s-1),-1),:B)^d


# given a vector `x`, a polynomial ρ, and a present time `t`
# Computes $\sum_i\rho_iB^ix_t := \sum_i\rho_ix_{t-i}$
# The computation is done as
# $$\left [ x_{t}, ..., x_{t-n} \right ] \cdot \left [ \rho_{0}, ..., \rho_{n} \right ]$$
function backwarded_sum(x,ρ::P,t) where P <: Polynomial

    if length(ρ) >= 1 # we check that there are some coefficients in the polynomial, otherwise this has no sense
        return backshifted_view(x,t + 1,Base.oneto(length(ρ))) ⋅ coeffs(ρ)
    else
        return zero(eltype(x))
    end

end

function backwarded_sum(x,ρ::A,t) where A <: AbstractArray

    if length(ρ) >= 1 # we check that there are some coefficients in the polynomial, otherwise this has no sense
        return backshifted_view(x,t + 1,Base.oneto(length(ρ))) ⋅ ρ
    else
        return zero(eltype(x))
    end

end

# given a vector `v = [a,b,c,...]` and a seasonality `s`
# produces a vector `[0₁, ..., 0ₛ₋₁,a,0₁, ..., 0ₛ₋₁,b,0₁, ..., 0ₛ₋₁,c,0₁, ..., 0ₛ₋₁,...]`
function seasonal_vector(v::Vector{T},s) where T
    T.(vcat(eachrow(hcat(zeros(eltype(v),length(v),s-1),v))...))
end

function seasonal_vector(v::SVector{N,T},s) where N,T
    T.(vcat(eachrow(hcat(zeros(eltype(v),length(v),s-1),v))...))
end

# given a vector `x = [x₁,x₂,x₃,...]` of coefficients (for the ar, ma, ax, ... component),
# a seasonality `s`
# builds the polynomial $1+\sum_i xᵢB^{s*i}$
function c2p(x::Vector{T},s) where T
    if isempty(x)
        return Polynomial([1])
    else
        return Polynomial(vcat(one(T),seasonal_vector(x,s)),:B)
    end
end