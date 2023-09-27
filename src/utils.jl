# this function allows to index into a vector with zero or negative indices
# we use it when we build x[t] as a sum of elements x[t-k]
# and it allow us not to have to check whether k >= t
function getidx(v::V,i::Int) where V<:Vector{T} where T<:Number
    @inbounds vᵢ = i > 0 ? v[i] : zero(T)
    return vᵢ
end

# extension of basic getidx to array of indices
getidx(v,Idx::I) where I<:AbstractArray  = [getidx(v,i) for i in Idx]

lag_access(x,t,l) = getidx(x,t .- l)

Uno = Polynomial([1],:B)

Δ(; d) = Polynomial(Int8[1,-1],:B)^d
Δ(; s = 1,d) = Polynomial(vcat(1,zeros(Int8,s-1),-1),:B)^d

function pushedback_sum(x,poly,t)
    @assert length(poly) >= 1

    return lag_access(x,t,1:length(poly))' * coeffs(poly)
end

function laggedvector(v,s)
    vcat(eachrow(hcat(zeros(eltype(v),length(v),s-1),v))...)
end

function coeffpoly(v,s)
    Polynomial(vcat(1,laggedvector(v,s)),:L)
end