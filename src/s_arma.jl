@kwdef struct Sarma{T<:Real}
    p::Int = 1
    q::Int = 1
    s::Int = 1
    ar::Vector{T} = [0.5]
    ma::Vector{T} = [0.5]
    dₙ = Normal(zero(T),one(T))
end

# Simulate ARMA process
function simulate_arma(p::Int, q::Int, ar::Vector{T}, ma::Vector{T}, n::Int, dₙ::D; s::Int = 1) where {T, D<:Distribution}
    
    # initialize a place holder for x
    x = Array{T, 1}(undef, n)

    # sample n random observations from the noise probability distribution, dₙ
    z = rand(dₙ,n)
    
    # we iterate over the series x to add the effects of the past
    lagger!(x,z,p,q,s,ar,ma)
    
    return x
end

# Simulate ARMA process with given generative noise
function simulate_arma(p::Int, q::Int, ar::Vector{T}, ma::Vector{T}, n::Int, dₙ::D; s::Int = 1) where {T, D<:Array}
    
    # check that dₙ is of the right length
    @assert (first∘size)(dₙ) = n

    # initialize a place holder for x
    x = similar(dₙ)

    # we iterate over the series x to add the effects of the past
    lagger!(x,dₙ,p,q,s,ar,ma)
    
    return x
end

realise(params::Sarma,n) = simulate_arma(params.p, params.q, params.ar, params.ma, n, params.dₙ; s = params.s)

# Simulate SARIMA with multiple seasonal components
function realise(X::Vector{Sarma{T}},n::Int) where T <: Real
    combined = Array{T,1}(undef,n)
   
    combined .= sum(realise.(X,n))

    return combined
end

# incremental adding process
function lagger!(x,z,p,q,s,ar,ma)
    @turbo for t in 1:n
        x[t] = z[t]

        # AR process
        for i in 1:p
            lag = i * s # when s is 1 this is just i, so non-seasonal
            x[t] += (t > lag ? ar[i] * x[t-lag] : 0)
        end

        # MA process
        for i in 1:q
            lag = i * s
            x[t] += (t > lag ? ma[i] * z[t-lag] : 0)
        end
    end

    return x

end