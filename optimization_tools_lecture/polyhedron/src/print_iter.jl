using Printf

function print_iter(iter::Int, fval::Float64, grad_norm::Float64, stpsize=-Inf)
	"""Print overview of iteration to command line."""
	
	if iter == 0
		@printf("iter	obj		norm(grad)	stpsize\n")
		@printf("%3d	%.4e	%.4e	\n",  iter, fval, grad_norm)
	else
		@printf("%3d	%.4e	%.4e	%.4e\n",  iter, fval, grad_norm, stpsize)
	end

end

