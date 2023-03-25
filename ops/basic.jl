function (Base.:+)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> 1
	r_grad = (l, r) -> 1

	return Operation([x, y], size(x), (l, r) -> (Base.:+)(l, r), [l_grad, r_grad])
end

function (Base.:-)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> 1
	r_grad = (l, r) -> -1

	return Operation([x, y], size(x), (a, b) -> (Base.:-)(a, b), [l_grad, r_grad])
end

function (Base.:/)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> 1 ./ r
	r_grad = (l, r) -> -l .* r .^ -2

	return Operation([x, y], size(x), (a, b) -> (Base.:/)(a, b), [l_grad, r_grad])
end

function (Base.:^)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	l_grad = (l, r) -> r .* l .^ (r .- 1)
	r_grad = (l, r) -> l .^ r .* log.(l)

	return Operation([x, y], size(x), (a, b) -> (Base.:^)(a, b))
end


function (Base.:*)(x::Tensor{0}, y::Tensor)
	l_grad = (l, r) -> r
	r_grad = (l, r) -> l
	return Operation([x, y], size(y), (a, b) -> (Base.:*)(a, b), [l_grad, r_grad])
end

function (Base.:*)(x::Tensor, y::Tensor{0})
	return (Base.:*)(y, x)
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

	return Operation([x, y], new_size, (a, b) -> (Base.:*)(a, b), [l_grad, r_grad])
end
