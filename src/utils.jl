# this function allows to index into a vector with zero or negative indices
# we use it when we build x[t] as a sum of elements x[t-k]
# and it allow us not to have to check whether k >= t
function getidx(v::V,i::Int)::T where V<:AbstractArray{T} where T<:Number
    vᵢ = i > 0 ? v[i] : zero(T)
    return vᵢ
end

# extension of basic getidx to array of indices
getidx(v,Idx::I) where I<:AbstractArray  = [getidx(v,i) for i in Idx]
getidx(v,Idx::I) where I<:Base.OneTo{<:Number}  = [getidx(v,i) for i in Idx]

# given a vector `x`, an index `t` in `x` (the *current time*), and a vector of indexes `I`
# returns the backshifted view `x[t-i]` for `i ∈ l`
backshifted_view(x,t,I) = @inline getidx(x,t .- I)

# the constant polynomial $1$ in the variable `:B`
const One = Polynomial([1],:B)

# we define the Lag operator polynomial, as $(1-B^s)^d$
# this is the special case for `s = 1`
Δ(d) = Polynomial(Int[1,-1],:B)^d

# we define the Lag operator polynomial, as $(1-B^s)^d$
Δ(; s = 1,d) = Polynomial(vcat(1,zeros(Int,s-1),-1),:B)^d

# given a vector `x`, a vector of coefficients ρ, and a present time `t`
# Computes $\sum_i\rho_iB^ix_t := \sum_i\rho_ix_{t-i}$
# The computation is done as
# $$\left [ x_{t}, ..., x_{t-n} \right ] \cdot \left [ \rho_{0}, ..., \rho_{n} \right ]$$
function backwarded_sum(x,ρ::A,t,buf = default_buffer()) where A <: AbstractArray
    if length(ρ) >= 1 # we check that there are some coefficients in the polynomial, otherwise this has no sense
        @no_escape buf begin
            y = alloc(eltype(x),buf,length(ρ))
            y .= backshifted_view(x,t + 1,Base.oneto(length(ρ)))
            y ⋅ ρ
        end
    else
        zero(eltype(x))
    end
end
backwarded_sum(arrey_param::ArrayParams,t,buf = default_buffer()) = backwarded_sum(arrey_param.vector,arrey_param.coefficients,t)
backwarded_sum(arreys_params::Vector{ArrayParams},t) = sum(backwarded_sum.(arreys_params,t))


# given a vector `v = [a,b,c,...]` and a seasonality `s`
# produces a vector `[0₁, ..., 0ₛ₋₁,a,0₁, ..., 0ₛ₋₁,b,0₁, ..., 0ₛ₋₁,c,0₁, ..., 0ₛ₋₁,...]`
function seasonal_vector(v::Vector{T},s) where T
    sv = zeros(T,length(v)*s)
    if s > 0 && length(v) > 0
        sv[s:s:end] .= v
    end
    return sv
end

# given a vector `v = [a,b,c,...]` and a seasonality `s`
# produces a vector `[0₁, ..., 0ₛ₋₁,a,0₁, ..., 0ₛ₋₁,b,0₁, ..., 0ₛ₋₁,c,0₁, ..., 0ₛ₋₁,...]`
# (or just identica to `v` if `s` is 1)
function seasonal_vector(v::SVector{N,T},s) where {N,T}
    sv = zeros(T,N*s)
    if s > 0 && N > 0
        sv[s:s:end] .= v
    end
    return sv
end

function get_z(dₙ, n)

    if typeof(dₙ) <: Distribution
        z = rand(dₙ,n)
    elseif typeof(dₙ) <: Array
        z = dₙ
    else
        error("The processes needs either a white noise sequence of a distribution with which to generate it. Instead, it received an object of type " * string(typeof(dₙ)))
    end

    if length(z) < 90
        z = SVector{length(z)}(z)
    end

    return z

end