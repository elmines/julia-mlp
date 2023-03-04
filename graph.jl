
export InputDict, forward

InputDict = Dict{Tensor, Union{Array{<:Number}, <:Number}}

function raw_forward(x::Parameter, inputs::InputDict)::ConcreteTensor
	raw_val = inputs[x]
	if size(raw_val) != size(x)
		throw(DimensionMismatch(""))
	end
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
	for (tensor, evaluation) in inputs
		if size(tensor) != size(evaluation)
			throw(DimensionMismatch("Concrete size " * size(evaluation) * " doesn't match graph size " * size(tensor)))
		end
	end
	return raw_forward(x, inputs)
end
