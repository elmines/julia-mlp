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

