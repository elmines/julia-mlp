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
@overload_binary_op(Base.:*)
@overload_binary_op(Base.:/)
@overload_binary_op(Base.:^)

