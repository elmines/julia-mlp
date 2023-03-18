
export InputDict, forward, Model

InputDict = Dict{Tensor, Union{Array{<:Number}, <:Number}}

function raw_forward(x::Input, inputs::InputDict)
	raw_val = inputs[x]
	#if size(raw_val) != size(x)
	#	throw(DimensionMismatch(""))
	#end
	return raw_val
end

function raw_forward(x::Operation, inputs::InputDict)
	callbackArgs = [raw_forward(parent, inputs) for parent in x.parents]
	return (x.callback)(callbackArgs...)
end

function raw_forward(x::Constant, inputs::InputDict)
	return x.value
end

function raw_forward(x::Parameter, inputs::InputDict)
	return x.value
end

function forward(x::Tensor, inputs::InputDict)
	#for (tensor, evaluation) in inputs
	#	if size(tensor) != size(evaluation)
	#		throw(DimensionMismatch("Concrete size $(size(evaluation)) doesn't match graph size $(size(tensor))"))
	#	end
	#end
	return raw_forward(x, inputs)
end

struct Model{NumIn, NumOut}
	inputs::Vector{<:Input}
	outputs::Vector{<:Tensor}
	trainables::Vector{<:Parameter}
	objective::Union{Tensor{0}, Missing}
	Model(inputs::Vector{<:Input}, outputs::Vector{<:Tensor}, trainables::Vector{<:Parameter}, objective::Union{Tensor{0}, Missing}=missing) = new{length(inputs), length(outputs)}(inputs, outputs, trainables, objective)
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

function Model(inputs::Vector{<:Input}, outputs::Vector{<:Tensor}, objective::Union{Tensor{0}, Missing} = missing)
	_validate_graph(inputs, outputs)
	if ismissing(objective)
		params = Vector{Parameter}()
	else
		params = _validate_graph(inputs, [objective], true)
	end
	return Model(inputs, outputs, params, objective)
end

