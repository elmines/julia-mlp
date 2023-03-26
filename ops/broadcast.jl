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

function Base.Broadcast.broadcasted(::TensorStyle, ::typeof(Base.:+), x, y)
	x::Tensor = make_tensor(x)
	y::Tensor = make_tensor(y)
	new_axes = Base.Broadcast.combine_axes(x, y)
	new_size = Tuple(length(ax) for ax in new_axes)

	l_grad = (l, r) -> 1
	r_grad = (l, r) -> 1

	return Operation([x, y], new_size, (a, b) -> broadcast(Base.:+, a, b), [l_grad, r_grad], "Add")
end

function Base.Broadcast.broadcasted(::TensorStyle, ::typeof(Base.:-), x, y)
	x::Tensor = make_tensor(x)
	y::Tensor = make_tensor(y)
	new_axes = Base.Broadcast.combine_axes(x, y)
	new_size = Tuple(length(ax) for ax in new_axes)

	l_grad = (l, r) -> 1
	r_grad = (l, r) -> -1
	return Operation([x, y], new_size, (a, b) -> broadcast(Base.:-, a, b), [l_grad, r_grad], "Sub")
end

function Base.Broadcast.broadcasted(::TensorStyle, ::typeof(Base.:/), x, y)
	x::Tensor = make_tensor(x)
	y::Tensor = make_tensor(y)
	new_axes = Base.Broadcast.combine_axes(x, y)
	new_size = Tuple(length(ax) for ax in new_axes)

	l_grad = (l, r) -> 1 ./ r
	r_grad = (l, r) -> -l .* r .^ -2
	return Operation([x, y], new_size, (a, b) -> broadcast(Base.:/, a, b), [l_grad, r_grad], "Div")
end

function Base.Broadcast.broadcasted(::TensorStyle, ::typeof(Base.:*), x, y)
	x::Tensor = make_tensor(x)
	y::Tensor = make_tensor(y)
	new_axes = Base.Broadcast.combine_axes(x, y)
	new_size = Tuple(length(ax) for ax in new_axes)

	l_grad = (l, r) -> r
	r_grad = (l, r) -> l
	return Operation([x, y], new_size, (a, b) -> broadcast(Base.:*, a, b), [l_grad, r_grad], "Mul")
end

function Base.Broadcast.broadcasted(::TensorStyle, ::typeof(Base.:^), x, y)
	x::Tensor = make_tensor(x)
	y::Tensor = make_tensor(y)
	new_axes = Base.Broadcast.combine_axes(x, y)
	new_size = Tuple(length(ax) for ax in new_axes)

	l_grad = (l, r) -> r .* l .^ (r .- 1)
	r_grad = (l, r) -> l .^ r .* log.(l)
	return Operation([x, y], new_size, (a, b) -> broadcast(Base.:^, a, b), [l_grad, r_grad], "Pow")
end

