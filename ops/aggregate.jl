
function Base.sum(x::Tensor)::Tensor{0}
	return Operation([x], (), a -> sum(a))
end
