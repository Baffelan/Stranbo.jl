function mixed_dirac_normal(σₖ,p)
    distro = MixtureModel(
    Normal[
        Normal(0, 0), # for measure theoretical issues we define the dirac as normal with variance 0
        Normal(0.0, σₖ)],
        [1-p,p]
    )

    return distro

end

realise(a::T,n) where T <: Distribution = rand(a,n)