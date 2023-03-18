#!/usr/bin/env julia

include("minesnet.jl")

using .MinesNet

# x has 60 dimensions
# y has 20 dimensions
function f(x)
	return 50 * sin(x[begin:20]) / sqrt(x[40:end]) + (x[10:30]^3 * cos(x[35:55]) - exp(x[40:end] / 10))^2
end

function apply_dense(x::Tensor, output_size; activation=missing, use_bias=false)
	n_features = size(x)[end]
	W = Parameter((n_features, output_size))

	h = x * W
	if use_bias
		b = Parameter((output_size,))
		h = h + b
	end
	if !ismissing(activation)
		h = activation(h)
	end
	return h
end

features = Input((60,))
labels = Input((20,))
h1 = apply_dense(features, 40; activation=tanh, use_bias=true)
output = apply_dense(features, 20, use_bias=true)
loss = (output - labels) .^ 2.0
loss = sum(loss)
@show loss
