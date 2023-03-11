# Broadcasting
struct TensorStyle <: Base.BroadcastStyle end

function Base.BroadcastStyle(::Type{<:Tensor})
	return TensorStyle()
end

function Base.BroadcastStyle(::TensorStyle, ::Base.BroadcastStyle)
	return TensorStyle()
end

function Broadcast.broadcastable(x::Tensor)
	return x
end

macro overload_binary_op(op)
	return quote
                ($op)(x::Number, y::Tensor) = ($op)(Constant(x), y)
		($op)(x::Tensor, y::Number) = ($op)(x, Constant(y))
                ($op)(x::Array, y::Tensor) = ($op)(Constant(x), y)
		($op)(x::Tensor, y::Array) = ($op)(x, Constant(y))
	end
end
@overload_binary_op(Base.:+)
@overload_binary_op(Base.:-)
@overload_binary_op(Base.:/)
@overload_binary_op(Base.:^)

macro define_binary_broadcast(op)
	return quote
		function Base.Broadcast.broadcasted(::TensorStyle, ::typeof($op), x::Tensor, y::Tensor)
			new_axes = Base.Broadcast.combine_axes(axes(x), axes(y))
			new_size = Tuple(length(ax) for ax in new_axes)
			return Operation([x, y], new_size, (a, b) -> broadcast($op, a, b))
		end

		function Broadcast.broadcasted(::TensorStyle, ::typeof($op), x, y::Tensor)
			return broadcast($op, Constant(x), y)
		end

		function Broadcast.broadcasted(::TensorStyle, ::typeof($op), x::Tensor, y)
			return broadcast($op, x, Constant(y))
		end
	end
end
@define_binary_broadcast(Base.:+)
@define_binary_broadcast(Base.:-)
@define_binary_broadcast(Base.:/)
@define_binary_broadcast(Base.:^)
@define_binary_broadcast(Base.:*)


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


