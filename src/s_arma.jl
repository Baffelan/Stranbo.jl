@kwdef struct SARIMA{T<:Real}
    s::Int
    d::Int
    ar::Vector{T}
    ma::Vector{T}
    dₙ
end

function  sarima(; T::Type{<:Real} = Float64,  s::Int = 1, d::Int = 0, ar = T[], ma = T[], dₙ = Normal(zero(T),one(T)))
    SARIMA{T}(s,d,ar,ma,dₙ)
end

# Alias for Simulate SARMA with multiple seasonal components
function realise(X::Vector{SARIMA{T}},n::Int,dₙ) where T <: Real
    return sample(X::Vector{SARIMA{T}},n::Int; dₙ = dₙ)
end

# Simulate ARIMA process
function sample(sarima::SARIMA{T},n::Int) where T  <: Real
    
    (; s, d, ar, ma, dₙ) = sarima

    # initialize a place holder for x

    # sample n random observations from the noise probability distribution, dₙ
    if typeof(dₙ) <: Distribution
        z = rand(dₙ,n)
    elseif typeof(dₙ) <: Array
        z = dₙ
    else error("The processes needs either a white noise sequence of a distribution with which to generate it. Instead, it received an object of type " * string(typeof(dₙ)))
    end
    # initialize a place holder for x  
    x = zeros(T,n)
    
    # create the lag polynomials with the corresponding coefficients
    x_poly = One - Polynomial([1,-seasonal_vector(ar, s)...],:B) * Δ(s = s, d = d)
    
    z_poly = Polynomial([1, seasonal_vector(ma, s)...],:B)
    
    # we iterate over the series x to add the effects of the past
    lagger!(x,z,x_poly,z_poly)

    return x
end

function sample(V::Vector{SARIMA{T}},n::Int; dₙ = nothing) where T <: Real
    
    # if dₙ is not define by user, the dₙ in the first sarima will be used
    if isnothing(dₙ)
        dₙ = V[1].dₙ
    end

    # sample n random observations from the noise probability distribution, dₙ
    if typeof(dₙ) <: Distribution
        z = rand(dₙ,n)
    elseif typeof(dₙ) <: Array
        z = dₙ
    else error("The processes needs either a white noise sequence of a distribution with which to generate it. Instead, it received an object of type " * string(typeof(dₙ)))
    end
    # initialize a place holder for x  
    x = zeros(T,n)

    # create the lag polynomials with the corresponding coefficients
    poly_ar = prod([Polynomial([1,-seasonal_vector(p.ar, p.s)...], :B) for p in V if !isempty(p.ar)])
    Δᵥ      = prod([Δ(s = p.s, d = p.d) for p in V if !iszero(p.d)])
    x_poly = One - poly_ar*Δᵥ
    
    z_poly = prod([Polynomial([1, seasonal_vector(p.ma, p.s)...], :B) for p in V if !isempty(p.ma)])

    # we iterate over the series x to add the effects of the past
    lagger!(x,z,x_poly,z_poly)

    return x
end

# incremental adding process
function lagger!(x,z,poly_ar,poly_ma)
    @show poly_ar
    @show poly_ma
    for t in 1:length(x)
    @inbounds  x[t] = backwarded_sum(x,poly_ar,t) +
                      backwarded_sum(z,poly_ma,t)
    end

    return x

end