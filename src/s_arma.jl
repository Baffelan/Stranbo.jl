@kwdef struct SARIMA{T<:Real}
    s::Int
    d::Int
    ar::SV where {SV <: SVector{N,T} where N}
    ma::SV where {SV <: SVector{M,T} where M}
    dₙ
end

function  sarima(; T::Type{<:Real} = Float64,  s::Int = 1, d::Int = 0, ar = T[], ma = T[], dₙ = Normal(zero(T),one(T)))
    SARIMA{T}(s,d,SVector{length(ar)}(ar),SVector{length(ma)}(ma),dₙ)
end

# Alias for Simulate SARMA with multiple seasonal components
function realise(X::Vector{SARIMA{T}},n::Int,dₙ) where T <: Real
    return sample(X::Vector{SARIMA{T}},n::Int; dₙ = dₙ)
end

# Simulate ARIMA process
function sample(sarima::SARIMA{T},n) where T  <: Real
    
    (; s, d, ar, ma, dₙ) = sarima

    z = get_z(dₙ,n)

    # initialize a place holder for x  
    x = zeros(T,n)
    
    # create the lag polynomials with the corresponding coefficients
    x_poly = One - Polynomial([1,-seasonal_vector(ar, s)...],:B) * Δ(s = s, d = d)
    x_poly = SVector{length(x_poly)}(coeffs(x_poly))
    
    z_poly = Polynomial([1, seasonal_vector(ma, s)...],:B)
    z_poly = SVector{length(z_poly)}(coeffs(z_poly))


    # we iterate over the series x to add the effects of the past
    lagger!(x,z,x_poly,z_poly)

    return x
end

function sample(V::Vector{SARIMA{T}},n; dₙ = nothing) where T <: Real
    
    # if dₙ is not define by user, the dₙ in the first sarima will be used
    if isnothing(dₙ)
        dₙ = V[1].dₙ
    end

    # sample n random observations from the noise probability distribution, dₙ

    z = get_z(dₙ,n)
    # initialize a place holder for x  
    x = zeros(T,n)

    # create the lag polynomials with the corresponding coefficients
    poly_ar = prod([Polynomial([1,-seasonal_vector(p.ar, p.s)...], :B) for p in V if !isempty(p.ar)])
    Δᵥ      = prod([Δ(s = p.s, d = p.d) for p in V if !iszero(p.d)])
    x_poly = One - poly_ar*Δᵥ
    x_poly = SVector{length(x_poly)}(coeffs(x_poly))

    z_poly = prod([Polynomial([1, seasonal_vector(p.ma, p.s)...], :B) for p in V if !isempty(p.ma)])
    z_poly = SVector{length(z_poly)}(coeffs(z_poly))

    # we iterate over the series x to add the effects of the past
    lagger!(x,z,x_poly,z_poly)

    return x
end

# incremental adding process
function lagger!(x,z,poly_ar,poly_ma)
    for t in 1:length(x)
    @inbounds  x[t] = backwarded_sum(x,poly_ar,t) +
                      backwarded_sum(z,poly_ma,t)
    end

    return x

end