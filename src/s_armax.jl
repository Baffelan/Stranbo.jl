@kwdef struct SARIMAX{T<:Real}
    s::Int
    d::Int
    ar::SVN where {SVN <: SVector{N,T} where N}
    ma::SVM where {SVM <: SVector{M,T} where M}
    ax::SVX where {SVX <: SVector{X,T} where X}
    dₙ
    axₙ::A where {A <: AbstractArray{<:Number}}
end

function  sarimax(; T::Type{<:Real} = Float64,
    s::Int = 1, d::Int = 0,
    ar = T[], ma = T[], ax = T[],
    dₙ = Normal(zero(T),one(T)),
    axₙ = T[])

    SARIMAX{T}(s,d,SVector{length(ar)}(ar),SVector{length(ma)}(ma),SVector{length(ma)}(ax),dₙ,axₙ)

end

# Simulate single component SARIMAX process
function sample(sarimax::SARIMAX{T},n::Integer) where T  <: Real
    
    (; s, d, ar, ma, ax, dₙ, axₙ) = sarimax

    z = get_z(dₙ,n)

    # initialize a place holder for x  
    x = zeros(T,n)
    
    # create the lag polynomials with the corresponding coefficients
    x_poly = One - Polynomial([one(T),-seasonal_vector(ar, s)...],:B) * Δ(s = s, d = d)  # _moving to the right_ all the terms of the ar polynomial but the constant 1
    coeffs_ar = SVector{length(x_poly)}(coeffs(x_poly))
    
    z_poly = Polynomial([1, seasonal_vector(ma, s)...],:B)
    coeffs_ma = SVector{length(z_poly)}(coeffs(z_poly))

    ax_poly = Polynomial([1, seasonal_vector(ax, s)...],:B)
    coeffs_ax = SVector{length(z_poly)}(coeffs(ax_poly))


    # we iterate over the series x to add the effects of the past
    lagger!(x,z,axₙ,coeffs_ar,coeffs_ma,coeffs_ax)

    return x
end

# Simulate multiple components SARIMAX process
function sample(V::Vector{SARIMAX{T}},n::Integer; dₙ = nothing) where T <: Real
    
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
    x_coeffs = SVector{length(x_poly)}(coeffs(x_poly))

    z_poly = prod([Polynomial([one(T), seasonal_vector(p.ma, p.s)...], :B) for p in V])
    z_coeffs = ArrayParams(z,SVector{length(z_poly)}(coeffs(z_poly)))

    arr_coeffs = [z_coeffs,[ArrayParams(v.axₙ,v.ax) for v in V if (!isempty(v.ax) && !isnothing(v.axₙ))]...]

    # we iterate over the series x to add the effects of the past
    lagger!(x,x_coeffs,arr_coeffs)

    return x
end

# incremental adding process
function lagger!(x,z,ax,coeffs_ar,coeffs_ma,coeffs_ax)
    for t in 1:length(x)
    @inbounds  x[t] = backwarded_sum(x,coeffs_ar,t) + # ar component
                      backwarded_sum(z,coeffs_ma,t) + # ma component
                      backwarded_sum(ax,coeffs_ax,t)
    end

    return x

end

function lagger!(x,coeffs_x,arr_coeffs)
    for t in 1:length(x)
    @inbounds  x[t] = backwarded_sum(x,coeffs_x,t) + # ar component
                      backwarded_sum(arr_coeffs,t)
    end

    return x

end