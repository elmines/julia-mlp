
function Base.sum(x::Tensor)::Tensor{0}
	grad_callback::Function = a -> 1
	grad_callbacks = Vector{Function}()
	push!(grad_callbacks, grad_callback)
	return Operation([x], (), a -> sum(a), grad_callbacks)
end
