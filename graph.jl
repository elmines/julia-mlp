
export InputDict, forward, Model

InputDict = Dict{Tensor, Union{Array{<:Number}, <:Number}}

function raw_forward(x::Parameter, inputs::InputDict)::ConcreteTensor
	raw_val = inputs[x]
	#if size(raw_val) != size(x)
	#	throw(DimensionMismatch(""))
	#end
	return raw_val
end

function raw_forward(x::Operation, inputs::InputDict)::ConcreteTensor
	callbackArgs::Vector{ConcreteTensor} = [raw_forward(parent, inputs) for parent in x.parents]
	return (x.callback)(callbackArgs...)
end

function raw_forward(x::Constant, inputs::InputDict)::ConcreteTensor
	return x.value
end

function forward(x::Tensor, inputs::InputDict)::ConcreteTensor
	#for (tensor, evaluation) in inputs
	#	if size(tensor) != size(evaluation)
	#		throw(DimensionMismatch("Concrete size $(size(evaluation)) doesn't match graph size $(size(tensor))"))
	#	end
	#end
	return raw_forward(x, inputs)
end

struct Model{NumIn, NumOut}
	inputs::Vector{<:Parameter}
	outputs::Vector{<:Tensor}
	trainables::Vector{<:Parameter}
	Model(inputs::Vector{<:Parameter}, outputs::Vector{<:Tensor}, trainables::Vector{<:Parameter}) = new{length(inputs), length(outputs)}(inputs, outputs, trainables)
end

function get_trainables(x::{<:Operation})::Set{<:Parameter}
	train_set::Set{<:Parameter} = Set()
	queue = Vector{<:Tensor}(x.parents)

	while length(queue) > 0
		y = pop!(queue)
		if isa(y, Operation)
			push!(queue, y.parents...)
		elseif isa(y, Parameter) && y.trainable
			push!(train_set, y)
		end
	end
	return train_set
end

function Model(inputs::Vector{<:Parameter}, outputs::Vector{<:Tensor})
	trainables = Set()
	visited = Set()

	queue = Vector{<:Tensor}(outputs)
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
		elseif isa(node, Parameter)
			push!(trainables, node)
		end

	end
	return Model(inputs, outputs, trainables)
end
