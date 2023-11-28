

using Printf

function print_iter(iter, fval, grad_norm, stpsize=-Inf)
	"""Print overview of iteration to 
	command line.
	"""
	
	@printf("Number of objective evaluations: %d\n": nfct)
	@printf("Number of gradient evaluations: %d\n": ngrad)
	@printf("Number of Hessian evaluations: %d\n": nhess)

end

