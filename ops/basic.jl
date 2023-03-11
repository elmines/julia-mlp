function (Base.:+)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	return Operation([x, y], size(x), (a, b) -> (Base.:+)(a, b))
end

function (Base.:-)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	return Operation([x, y], size(x), (a, b) -> (Base.:-)(a, b))
end

function (Base.:/)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	return Operation([x, y], size(x), (a, b) -> (Base.:/)(a, b))
end

function (Base.:^)(x::Tensor, y::Tensor)
	if size(x) != size(y)
		throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
	end

	return Operation([x, y], size(x), (a, b) -> (Base.:^)(a, b))
end

