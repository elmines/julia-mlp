
function Base.sum(x::Tensor; name::String = "Sum")::Tensor{0}
	grad_callback::Function = a -> 1
	grad_callbacks = Vector{Function}()
	push!(grad_callbacks, grad_callback)
	return Operation([x], (), a -> sum(a), grad_callbacks, name)
end
