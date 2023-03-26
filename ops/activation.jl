
function Base.tanh(x::Tensor{N}; name::String = "Tanh")::Tensor{N} where N
	grad_callback::Function = a -> sech.(a) .^ 2
	grad_callbacks = Vector{Function}()
	push!(grad_callbacks, grad_callback)
	return Operation([x], size(x), a -> tanh.(a), grad_callbacks, name)
end

