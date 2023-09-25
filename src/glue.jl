function realise(X::Vector{Any},n::Int)

    combined = sum(realise.(X,n))

    return combined
end