# the function `realise_all` tries to apply `realise` to each component in the vector that is fed
# and summarises the output 
function realise_all(X::Vector{Any},n::Int)

    combined = sum(realise.(X,n))

    return combined
end

# Alias for Simulate SARIMA when run together with other additional noises
realise(X::Vector{SARIMA{T}},n::Int) where T <: Real = sample(X::Vector{SARIMA{T}},n::Int)