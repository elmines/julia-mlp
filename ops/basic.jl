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
function __inner_prod_grad_right(x::Array{T, 2}, y::Array{U, 2}) where {T, U}
	result = zeros(size(x)[begin], size(y)[end], size(y)...)
	left_axis = axes(x)[begin]
	right_axis = axes(y)[end]
	inner_axis = axes(x)[end]
	for l in right_axis
		recipient = result[left_axis, l, inner_axis, l]
		donor = x
		#@show size(recipient)
		#@show size(donor)
		recipient .= donor
	end
	return result
end

# Take the gradient w.r.t. the left tensor
function __inner_prod_grad_left(x::Array{T, 2}, y::Array{U, 2}) where {T, U}
	result = zeros(size(x)[begin], size(y)[end], size(x)...)
	left_axis = axes(x)[begin]
	right_axis = axes(y)[end]
	inner_axis = axes(x)[end]

	for j in left_axis
		recipient = result[j, right_axis, j, inner_axis]
		donor = transpose(y)
		#@show size(recipient)
		#@show size(donor)
		recipient .= donor
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

	return Operation([x, y], new_size, (a, b) -> (Base.:*)(a, b), [__inner_prod_grad_left, __inner_prod_grad_right], "Mul")
end
