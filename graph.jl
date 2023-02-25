
export InputDict, forward

InputDict = Dict{Tensor, Union{Array{<:Number}, <:Number}}

function forward(x::Parameter, inputs::InputDict)::Array{<:Number}
	raw_val = inputs[x]
	if size(raw_val) != size(x)
		throw(DimensionMismatch(""))
	end
	return raw_val
end

function forward(x::Operation, inputs::InputDict)::Array{<:Number}
	callbackArgs::Vector{Array{<:Number}} = [forward(parent, inputs) for parent in x.parents]
	return (x.callback)(callbackArgs...)
end

function forward(x::Constant, inputs::InputDict)::Array{<:Number}
	return x.value
end

