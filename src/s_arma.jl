@kwdef struct SARIMA{T<:Real}
    s::Int
    d::Int
    ar::SVN where {SVN <: SVector{N,T} where N}
    ma::SVM where {SVM <: SVector{M,T} where M}
    dₙ
end

function  sarima(; T::Type{<:Real} = Float64,  s::Int = 1, d::Int = 0, ar = T[], ma = T[], dₙ = Normal(zero(T),one(T)))
    SARIMA{T}(s,d,SVector{length(ar)}(ar),SVector{length(ma)}(ma),dₙ)
end

# Simulate ARIMA process
function sample(sarima::SARIMA{T},n::Integer) where T  <: Real
    
    (; s, d, ar, ma, dₙ) = sarima

    z = get_z(dₙ,n)

    # initialize a place holder for x  
    x = zeros(T,n)
    
    # create the lag polynomials with the corresponding coefficients
    x_poly = One - Polynomial([one(T),-seasonal_vector(ar, s)...],:B) * Δ(s = s, d = d)  # _moving to the right_ all the terms of the ar polynomial but the constant 1
    coeffs_ar = SVector{length(x_poly)}(coeffs(x_poly))
    
    z_poly = Polynomial([1, seasonal_vector(ma, s)...],:B)
    coeffs_ma = SVector{length(z_poly)}(coeffs(z_poly))


    # we iterate over the series x to add the effects of the past
    lagger!(x,z,coeffs_ar,coeffs_ma)

    return x
end

function sample(V::Vector{SARIMA{T}},n::Integer; dₙ = nothing) where T <: Real
    
    # if dₙ is not define by user, the dₙ in the first sarima will be used
    if isnothing(dₙ)
        dₙ = V[1].dₙ
    end

    # sample n random observations from the noise probability distribution, dₙ

    z = get_z(dₙ,n)
    # initialize a place holder for x  
    x = zeros(T,n)

    # create the lag polynomials with the corresponding coefficients
    poly_ar = prod([Polynomial([one(T),-seasonal_vector(p.ar, p.s)...], :B) for p in V])
    Δᵥ      = prod([Δ(s = p.s, d = p.d) for p in V])
    x_poly = One - poly_ar*Δᵥ # _moving to the right_ all the terms of the ar polynomial but the constant 1
    coeffs_ar = SVector{length(x_poly)}(coeffs(x_poly))

    z_poly = prod([Polynomial([one(T), seasonal_vector(p.ma, p.s)...], :B) for p in V])
    coeffs_ma = SVector{length(z_poly)}(coeffs(z_poly))

    # we iterate over the series x to add the effects of the past
    lagger!(x,z,coeffs_ar,coeffs_ma)

    return x
end

# incremental adding process
function lagger!(x,z,coeffs_ar,coeffs_ma)
    for t in eachindex(x)
    @inbounds  x[t] = backwarded_sum(x,coeffs_ar,t) + # ar component
                      backwarded_sum(z,coeffs_ma,t) # ma component
    end

    return x

end