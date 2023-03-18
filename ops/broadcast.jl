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
		function Base.Broadcast.broadcasted(::TensorStyle, ::typeof($op), x, y)
			x::Tensor = make_tensor(x)
			y::Tensor = make_tensor(y)
			new_axes = Base.Broadcast.combine_axes(x, y)
			new_size = Tuple(length(ax) for ax in new_axes)
			return Operation([x, y], new_size, (a, b) -> broadcast($op, a, b))
		end
	end
end
@define_binary_broadcast(Base.:+)
@define_binary_broadcast(Base.:-)
@define_binary_broadcast(Base.:/)
@define_binary_broadcast(Base.:*)
@define_binary_broadcast(Base.:^)

