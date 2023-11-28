using Printf

include("armijo.jl")
include("print_iter.jl")


function gradmethod(nlp, x::Vector{Float64};
		tol::Float64=1e-2, itmax::Int64=500,
		intermediate_callback=Union{Function,Nothing}=Nothing)

	print_gradmethod(length(x))

	it = 0
	nfct = 0
	ngrad = 0
	y = deepcopy(x)

	fval = obj(nlp,x)
	g = grad(nlp, x)
	normg = norm(g, 2)
	nfct += 1
	ngrad += 1
	print_iter(it, fval, normg)

	while (normg>tol)&&(it<itmax)
		it += 1
		y, sig, fval, nfct = armijo(nlp, y, -g, g, fval, nfct)
		g = grad(nlp, y)
		normg = norm(g, 2)
		ngrad += 1
		print_iter(it, fval, normg, sig)

		if isa(intermediate_callback, Function)
			intermediate_callback(y, it)
		end

	end

	@printf("Number of objective evaluations: %d\n",  nfct)
	@printf("Number of gradient evaluations: %d\n", ngrad)

	if it >= itmax
		@printf("Maximum number of iterations (%d) reached.\n", it)
	end

	return y
end

"""gradmethod(x::Vector{Float64}, fct::Function, grad::Function;
		tol::Float64=1e-6, itmax::Int64=50) -> Vector{Float64}

Gradient descent method with Armijo line search.

The method requires an initial point `x`, the function
`fct` to be minimized as well as its gradient `grad`. 

The user may alter the termination tolerance `tol`
and the maximum number of iterations `itmax` preformed. 

The method terminates if `norm(grad fct(x))) <= tol`.
Here, `norm` is the Eucleadian norm.
"""


function gradmethod(x::Vector{Float64}, fct::Function, grad::Function;
		tol::Float64=1e-3, itmax::Int64=2500)

	print_gradmethod(length(x))

	it = 0
	nfct = 0
	ngrad = 0
	y = deepcopy(x)

	fval = fct(x)
	g = grad(x)
	normg = norm(g, 2)
	nfct += 1
	ngrad += 1
	print_iter(it, fval, normg)

	while (normg>tol)&&(it<itmax)
		it += 1
		y, sig, fval, nfct =armijo(y, -g, fct, g, fval, nfct)
		g = grad(y)
		normg = norm(g, 2)
		ngrad += 1
		print_iter(it, fval, normg, sig)
	end

	@printf("Number of objective evaluations: %d\n",  nfct)
	@printf("Number of gradient evaluations: %d\n", ngrad)

	if it >= itmax
		@printf("Maximum number of iterations (%d) reached.\n", it)
	end

	return y
end


function print_gradmethod(nvar::Int64)

	@printf("#############################################################\n")
	@printf("\n")
	@printf("Gradient method for unconstrained minimization problems.\n")
	@printf("\n")
	@printf("Number of optimization variables: %d\n", nvar)
	@printf("\n")
	@printf("#############################################################\n")

end
