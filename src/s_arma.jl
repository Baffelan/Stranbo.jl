@kwdef struct Sarma{T<:Real}
    s::Int = 1
    ar::Vector{T} = []
    ma::Vector{T} = []
    dₙ = Normal(zero(T),one(T))
end

@kwdef struct Sarima{T<:Real}
    s::Int8 = 1
    d::Int = 0
    ar::Vector{T} = []
    ma::Vector{T} = []
    dₙ = Normal(zero(T),one(T))
end

# Simulate ARMA process
function simulate_arma(SARMA::Sarma{T},n::Int) where T <: Number
    
    (; s, ar, ma, dₙ) = SARMA

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

    # create the lag polynomial with the corresponding coefficients
    poly_ar = c2p(ar,s) - One
    poly_ma = c2p(ma,s)

    # we iterate over the series x to add the effects of the past
    lagger!(x,z,poly_ar,poly_ma)

    return x
end

function simulate_arma(V::Vector{Sarma{T}},n::Int; dₙ = nothing) where T <: Real
    
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

    # create the lag polynomial with the corresponding coefficients
    poly_ar = prod([c2p(p.ar, p.s) for p in V if !isempty(p.ar)]) - One
    poly_ma = prod([c2p(p.ma, p.s) for p in V if !isempty(p.ma)])

    # we iterate over the series x to add the effects of the past
    lagger!(x,z,poly_ar,poly_ma)

    return x
end

# Alias for Simulate SARMA with multiple seasonal components
function realise(X::Vector{Sarma{T}},n::Int,dₙ) where T <: Real
    return simulate_arma(X::Vector{Sarma{T}},n::Int; dₙ = dₙ)
end

# Simulate ARIMA process
function simulate_arima(SARIMA::Sarima,n::Int)
    
    (; s, d, ar, ma, dₙ) = SARIMA

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
    
    # create the lag polynomial with the corresponding coefficients
    poly_ar = c2p(ar,s) - One
    poly_ma = c2p(ma,s)

    # we iterate over the series x to add the effects of the past
    lagger!(x,z,poly_ar,poly_ma)

    Δᵥ = One - Δ(s = s, d = d)

    for t in 1:length(x)
        @inbounds  x[t] = x[t] + backwarded_sum(x,Δᵥ,t)
    end
    
    return x
end

function simulate_arima(V::Vector{Sarima{T}},n::Int; dₙ = nothing) where T <: Real
    
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

    # create the lag polynomial with the corresponding coefficients
    poly_ar = prod([c2p(p.ar, p.s) for p in V if !isempty(sarma.ar)]) - One
    poly_ma = prod([c2p(p.ma, p.s) for p in V if !isempty(sarma.ma)])
    Δᵥ = One - prod([Δ(s = p.s, d = p.d) for p in V if !isempty(sarma.ma)])


    # we iterate over the series x to add the effects of the past
    lagger!(x,z,poly_ar,poly_ma)

    # o the integration
    for t in 1:length(x)
        @inbounds  x[t] = x[t] + backwarded_sum(x,Δᵥ,t)
    end

    return x
end

# incremental adding process
function lagger!(x,z,poly_ar,poly_ma)
  
    for t in 1:length(x)
    @inbounds  x[t] = backwarded_sum(x,poly_ar,t) +
                      backwarded_sum(z,poly_ma,t+1)
    end

    return x

end