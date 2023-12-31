# this function allows to index into a vector with zero or negative indices
# we use it when we build x[t] as a sum of elements x[t-k]
# and it allow us not to have to check whether k >= t
function getidx(v::Vector{T},i::Int)::T where T<:Number
    vᵢ = i > 0 ? v[i] : zero(T)
    return vᵢ
end

# extension of basic getidx to array of indices
function getidx(v::Vector{T},Idx::I) where {I<:AbstractArray,T<:Number}
    this_subv = Vector{T}(undef,length(I))
    
    for i in 1:length(Idx)
        @inbounds  this_subv[i] = getidx(v,Idx[i])
    end
    
    return this_subv
end

function getidx(v::Vector{T},Idx::I) where {I<:StepRange,T<:Number}
    this_subv = Vector{T}(undef,length(Idx))
    
    for i in 1:length(Idx)
    @inbounds  this_subv[i] = getidx(v,Idx[i])
    end

    return this_subv
end


function getidx(v::Vector{T},Idx::I) where {I<:Base.OneTo{<:Number},T<:Number}
    this_subv = Vector{T}(undef,length(I))
    
    for i in 1:length(Idx)
        @inbounds  this_subv[i] = getidx(v,Idx[i])
    end
    
    return this_subv
end

# given a vector `x`, an index `t` in `x` (the *current time*), and a vector of indexes `I`
# returns the backshifted view `x[t-i]` for `i ∈ l`
backshifted_view(x,t,I) = @inline  getidx(x,t .- I)

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
function backwarded_sum(x::Vector{T},ρ::SVector{N,T},t::Int)::T where {T <: Number, N}
    if N >= 1 # we check that there are some coefficients in the polynomial, otherwise this has no sense
        return allocated_back_dot(x,ρ,t+1)
    else
        return zero(T)
    end
end
function backwarded_sum(arrey_param::ArrayParams,t)
    x::eltype(arrey_param.vector) =  backwarded_sum(arrey_param.vector,arrey_param.coefficients,t)
    return x
end
backwarded_sum(arreys_params::Vector{ArrayParams},t) = sum(backwarded_sum.(arreys_params,t))

function allocated_back_dot(vector::Vector{T},coefficients::SVector{N,T},idx::Int; buff = default_buffer())::T where {T <: Number,N}
    @no_escape buff begin
        y = alloc(T,buff,N)
        y .= backshifted_view(vector,idx,Base.oneto(N))
        y ⋅ coefficients
    end
end


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

function get_z(dₙ::V, n) where V <: Vector
    return dₙ
end

function get_z(dₙ::D, n) where D <: Distribution
    return rand(dₙ,n)
end