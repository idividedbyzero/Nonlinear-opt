using DelimitedFiles
using PyPlot

include("../src/plotc60.jl")
include("../src/gradmethod.jl")
include("../src/GaussNewton.jl")
include("c60.jl")


# Read initial point
x = readdlm("/home/asdqwe123/phd_prep/Nonlinear-opt/optimization_tools_lecture/polyhedron/example/data/xinit1")
x = x[:, 1]; 


c60 = C60()

intermediate_callback(y, k) = plotc60(y, "gradmethod_k=$(k)")

gradmethod(c60, x; tol=1e-6, itmax=20, intermediate_callback=intermediate_callback)

intermediate_callback(y, k) = plotc60(y, "GaussNewton_k=$(k)")

GaussNewton(c60, x; tol=1e-6, itmax=20, intermediate_callback=intermediate_callback)


