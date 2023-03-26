function (Base.:+)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> 1
	r_grad = (l, r) -> 1

	return Operation([x, y], size(x), (l, r) -> (Base.:+)(l, r), [l_grad, r_grad], "Add")
end

function (Base.:-)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> 1
	r_grad = (l, r) -> -1

	return Operation([x, y], size(x), (a, b) -> (Base.:-)(a, b), [l_grad, r_grad], "Sub")
end

function (Base.:/)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> 1 ./ r
	r_grad = (l, r) -> -l .* r .^ -2

	return Operation([x, y], size(x), (a, b) -> (Base.:/)(a, b), [l_grad, r_grad], "Div")
end

function (Base.:^)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> r .* l .^ (r .- 1)
	r_grad = (l, r) -> l .^ r .* log.(l)

	return Operation([x, y], size(x), (a, b) -> (Base.:^)(a, b), "Pow")
end


function (Base.:*)(x::Tensor{0}, y::Tensor)
	l_grad = (l, r) -> r
	r_grad = (l, r) -> l
	return Operation([x, y], size(y), (a, b) -> (Base.:*)(a, b), [l_grad, r_grad], "Mul")
end

function (Base.:*)(x::Tensor, y::Tensor{0})
	return (Base.:*)(y, x)
end

# Take the gradient w.r.t. the right tensor
function __inner_prod_grad_right(x::Array, y::Array)
	new_size = (size(x)[begin:end-1]..., size(y)[begin+1:end]..., size(y)...)
	result = zeros(new_size...)
	left_axes = axes(result)[begin:ndims(x)]
	inner_axis = axes(x)[end]
	right_axes = axes(y)[2:end]

	for l in right_axes[begin]
		recipient = result[left_axes..., inner_axis, l]
		rvalue = x[left_axes...]
		recipient .= rvalue
	end
	return result
end

# Take the gradient w.r.t. the left tensor
function __inner_prod_grad_left(x::Array, y::Array)
	new_size = (size(x)[begin:end-1]..., size(y)[begin+1:end]..., size(y)...)
	result = zeros(new_size...)
	left_axes = axes(result)[begin:ndims(x)]
	inner_axis = axes(x)[end]
	right_axes = axes(y)[2:end]

	@show left_axes
	@show inner_axis
	@show right_axes

	for j in left_axes[begin]
		recipient = result[left_axes..., j]
		rvalue = y[inner_axis]
		@show size(recipient)
		@show size(rvalue)

		recipient .= rvalue
	end
	return result
end

function (Base.:*)(x::Tensor, y::Tensor)
	size_x = size(x)
	size_y = size(y)
	if size_x[end] != size_y[begin]
		throw(DimensionMismatch("Inner dimensions of operands for * do not match: " * string(size_x) * " vs. " * string(size_y)))
	end
	new_size = tuple(size_x[begin:end-1]..., size_y[begin+1:end]...)


	l_grad = (l, r) -> r
	r_grad = (l, r) -> l

	return Operation([x, y], new_size, (a, b) -> (Base.:*)(a, b), [__inner_prod_grad_right, __inner_prod_grad_left], "Mul")
end
