using LinearAlgebra
include("armijo.jl")
include("print_iter.jl")


function GaussNewton(nls, x::Vector{Float64}; 
		     tol::Float64=1e-6, itmax::Int64=20,
		     intermediate_callback::Union{Function, Nothing}=Nothing)
    

	it = 0
	F = residual(nls, x)	
	DF = jac_residual(nls, x)

	
	f = .5*dot(F, F); 
	g = transpose(DF)*F;
	y = copy(x)

    	normg = norm(g);

	print_iter(it, f, normg)
	
    	while (normg>tol) && (it<itmax)

		it += 1
		s = -DF\F

		y, sig, f = armijo(nls, y, s, g, f, 0)
		
		F = residual(nls, y)
		DF = jac_residual(nls, y)
		g = transpose(DF)*F 
		normg = norm(g)
	
		print_iter(it, f, normg, sig)
			
		if isa(intermediate_callback, Function)
			intermediate_callback(y, it)
		end

    	end
	
end



function GaussNewton(x::Vector{Float64}, fct::Function, grad::Function; 
		     tol::Float64=1e-6, itmax::Int64=20, bplot::Bool=false)
    
	it = 0
	F = fct(x)	
	DF = grad(x)

	
	fval = .5*dot(F, F); 
	g = transpose(DF)*F;
	y = copy(x)

    	normg = norm(g);

	function fct_(x)

		F = fct(x)
		return .5dot(F, F)
	end

	print_iter(it, fval, normg)
	
    	while (normg>tol) && (it<itmax)
		it += 1
		s = -DF\F

		y, sig, fval =armijo(y, s, fct_, g, fval,0)
		
		F = fct(y)
		DF = grad(y)
		f = 0.5*dot(F, F) 
		g = transpose(DF)*F 
		normg = norm(g)
	
		print_iter(it, fval, normg, sig)
			
		if bplot == true
			plotc60(y, "GaussNewton_k=$it")
		end

	
    	end
	
end
