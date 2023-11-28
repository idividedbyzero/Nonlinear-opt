
using ForwardDiff
using Random
using Test
include("../src/gradmethod.jl")

rng = MersenneTwister(1234)
x = randn(rng, 10)
println(x)
function f(x)

	return dot(x, x)

end

grad_f = x -> ForwardDiff.gradient(f, x)

y = gradverf(x, f, grad_f)

@test isapprox(y, zeros(length(y)))


function rosenbrock(x, a=1., b=100.)

	return (a-x[1])^2+b*(x[2]-x[1]^2)^2
end

grad_rosenbrock = x -> ForwardDiff.gradient(rosenbrock, x)

x = [2.0, 3.]


gradverf(x, rosenbrock, grad_rosenbrock)





