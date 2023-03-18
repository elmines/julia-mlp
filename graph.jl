
export Model, TensorDict, predict

TensorDict = Dict{Tensor, Union{<:Array{<:Number}, <:Number}}

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

struct Model{NumIn, NumOut}
	inputs::Vector{<:Input}
	outputs::Vector{<:Tensor}
	trainables::Vector{<:Parameter}
	labels::Union{<:Input, Missing}
	objective::Union{<:Tensor{0}, Missing}
	Model(inputs::Vector{<:Input}, outputs::Vector{<:Tensor}, trainables::Vector{<:Parameter}, labels::Union{<:Input, Missing}, objective::Union{<:Tensor{0}, Missing}) = new{length(inputs), length(outputs)}(inputs, outputs, trainables, labels, objective)
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
