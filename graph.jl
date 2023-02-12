
export InputDict, forward

InputDict = Dict{Tensor, Number}

function forward(x::Input, inputs::InputDict)::Number
	return inputs[x]
end

function forward(x::Operation, inputs::InputDict)::Number
	callbackArgs::Vector{Number} = [forward(parent, inputs) for parent in x.parents]
	return (x.callback)(callbackArgs...)
end

function forward(x::Constant, inputs::InputDict)::Number
	return x.value
end

function forward(x::Variable, inputs::InputDict)::Number
	return x.value
end

