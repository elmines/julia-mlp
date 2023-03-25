#!/usr/bin/env julia

include("minesnet.jl")

using .MinesNet
using Distributions:Uniform

# x has 60 dimensions
# y has 20 dimensions
function f(x)
	return 50 .* sin.(x[:, begin:20]) ./ cbrt.(x[:, 41:end]) .+ (x[:, 11:30].^3 .* cos.(x[:, 35:54]) .- exp.(x[:, 41:end] ./ 10)).^2
end

function apply_dense(x::Tensor, output_size::Int; activation=missing, use_bias=false)
	n_features = size(x)[end]
	W = Parameter((n_features, output_size))

	h = x * W
	if use_bias
		b = Parameter(zeros(size(x)[begin], output_size))
		h = h .+ b
	end
	if !ismissing(activation)
		h = activation(h)
	end
	return h
end

BATCH_SIZE = 2
NUM_FEATURES = 60
NUM_BATCHES = 1000

features = Input((BATCH_SIZE, NUM_FEATURES))
labels = Input((BATCH_SIZE, 20))
h1 = apply_dense(features, 40; activation=tanh, use_bias=true)
predictions = apply_dense(h1, 20, use_bias=true)
loss = (predictions - labels) .^ 2.0
loss = sum(loss)
@show loss

model = Model([features], [predictions], labels, loss)
input_dict = TensorDict(features => ones(Float64, size(features)))

distro = Uniform(-10, 10)
for _ in range(1, NUM_BATCHES)
	x = rand(distro, (BATCH_SIZE, NUM_FEATURES))
	y = f(x)
	batch_loss = fit(model, TensorDict(features => x), y)
	@show batch_loss
	break
end
