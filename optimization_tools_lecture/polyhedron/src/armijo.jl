using LinearAlgebra


"""Armijo line search method."""

function armijo(nlp, x::Vector{Float64}, s::Vector{Float64}, g::Vector{Float64},
		fval::Float64, nfct::Int64; gamma::Float64=0.01,
		beta::Float64=0.5)

	sigma = 1.0
    	y = x + s 
	fval_ = obj(nlp, y)
	nfct += 1
    	rhs = gamma*dot(g, s)

    	while fval_ - fval > sigma*rhs
		sigma *= beta
		y[:] = x+sigma*s
		fval_ = obj(nlp, y)
		nfct += 1
    	end
	
	return y, sigma, fval_, nfct

end

function armijo(x::Vector{Float64}, s::Vector{Float64}, 
		fct::Function, g::Vector{Float64},
		fval::Float64, nfct::Int64; gamma::Float64=0.01,
		beta::Float64=0.5)

	sigma = 1.0
    	y = x + s 
	fval_ = fct(y)
	nfct += 1
    	rhs = gamma*dot(g, s)

    	while fval_ - fval > sigma*rhs
		sigma *= beta
		y[:] = x+sigma*s
		fval_ = fct(y)
		nfct += 1
    	end
	
	return y, sigma, fval_, nfct

end

