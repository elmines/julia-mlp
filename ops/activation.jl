
function Base.tanh(x::Tensor{N})::Tensor{N} where N
	return Operation([x], size(x), a -> tanh.(a))
end
