
export Model, TensorDict, predict, fit

TensorDict = Dict{Tensor, Union{<:Array{<:Number}, <:Number}}
GradDict = Dict{Tuple{Tensor, Tensor}, ConcreteTensor}

function forward!(x::Input, cache::TensorDict)
	raw_val = cache[x]
	return raw_val
end

function forward!(x::Operation, cache::TensorDict)
	if !(x in keys(cache))
		callbackArgs = [forward!(parent, cache) for parent in x.parents]
		cache[x] = (x.callback)(callbackArgs...)
	end
	return cache[x]
end

function forward!(x::Constant, cache::TensorDict)
	return x.value
end

function forward!(x::Parameter, cache::TensorDict)
	return x.value
end

function forward!(tensors::Vector{<:Tensor}, cache::TensorDict)
	return [forward!(x, cache) for x in tensors]
end

function backward!(grad_dict::GradDict, x::Constant, parameters::Vector{<:Parameter}, cache::TensorDict)
	for param in parameters
		grad_dict[x, param] = 0.
	end
end

function backward!(grad_dict::GradDict, x::Input, parameters::Vector{<:Parameter}, cache::TensorDict)
	for param in parameters
		grad_dict[x, param] = 0.
	end
end

function backward!(grad_dict::GradDict, x::Parameter, parameters::Vector{<:Parameter}, cache::TensorDict)
	throw(ErrorException("This shouldn't be called."))
end


function backward!(grad_dict::GradDict, x::Operation, parameters::Vector{<:Parameter}, cache::TensorDict)
	callback_args = [cache[p] for p in x.parents] 
	for (parent, grad_callback) in zip(x.parents, x.grad_callbacks)
		if (x, parent) in keys(grad_dict)
			continue
		end
		if isa(parent, Parameter)
			grad_dict[x, parent] = grad_callback(callback_args...)
			continue
		end

		backward!(grad_dict, parent, parameters, cache)
		grad_dict[x, parent] = grad_callback(callback_args...)
		for param in parameters
			# Already computed this earlier
			if param in x.parents
				continue
			end

			if !((x, param) in keys(grad_dict))
				grad_dict[x, param] = 0.
			end
			grad_dict[x, param] .+= grad_dict[x, parent] .* grad_dict[parent, param]
		end

	end
end

struct Model{NumIn, NumOut}
	inputs::Vector{<:Input}
	outputs::Vector{<:Tensor}
	parameters::Vector{<:Parameter}
	labels::Union{<:Input, Missing}
	objective::Union{<:Tensor{0}, Missing}
	Model(inputs::Vector{<:Input}, outputs::Vector{<:Tensor}, parameters::Vector{<:Parameter}, labels::Union{<:Input, Missing}, objective::Union{<:Tensor{0}, Missing}) = new{length(inputs), length(outputs)}(inputs, outputs, parameters, labels, objective)
end

function _validate_graph(inputs::Vector{<:Input}, outputs::Vector{<:Tensor}, collect_params=false)::Vector{<:Parameter}
	params = Set{Parameter}()
	visited = Set{Tensor}()
	queue = Vector{Tensor}()
	push!(queue, outputs...)
	while length(queue) > 0
		node = pop!(queue)
		if node in visited
			continue
		end
		push!(visited, node)
		if isa(node, Operation)
			push!(queue, node.parents...)
		elseif isa(node, Input) && !(node in inputs)
			throw(ErrorException(string(node) * " was not listed in inputs"))
		elseif collect_params && isa(node, Parameter)
			push!(params, node)
		end
	end
	vector_params = Vector{Parameter}()
	push!(vector_params, params...)
	return vector_params
end

function Model(inputs::Vector{<:Input}, outputs::Vector{<:Tensor}, labels::Input, objective::Tensor{0})
	_validate_graph(inputs, outputs)
	params = _validate_graph([inputs..., labels], [objective], true)
	return Model(inputs, outputs, params, labels, objective)
end

function Model(inputs::Vector{<:Input}, outputs::Vector{<:Tensor})
	_validate_graph(inputs, outputs)
	return Model(inputs, outputs, Vector{Parameter}(), missing, missing)
end

function predict(model::Model, inputs::TensorDict)::Vector{ConcreteTensor}
	cache = TensorDict(k => v for (k,v) in inputs)
	return forward!(model.outputs, cache)
end

function fit(model::Model, x::TensorDict, y::ConcreteTensor; lr = 0.001)
	if ismissing(model.labels)
		throw(ErrorException("Tried to fit() a model with no objective"))
	end
	cache = TensorDict(k => v for (k, v) in x)
	cache[model.labels] = y
	forward!([model.outputs..., model.objective], cache)
	grad_dict = GradDict()
	backward!(grad_dict, model.objective, model.parameters, cache)

	for parameter in model.parameters
		parameter.value .-= grad_dict[model.objective, parameter] .* lr
	end

	return cache[model.objective]
end
