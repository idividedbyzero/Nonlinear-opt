using DelimitedFiles
using PyPlot
include("../src/plotc60.jl")
x = readdlm("/home/asdqwe123/phd_prep/Nonlinear-opt/optimization_tools_lecture/polyhedron/example/data/xopt"); x = x[:, 1]; 

plotc60(x, "solution")
