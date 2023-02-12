macro binary_op(op)
	return quote
		($op)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> ($op)(a, b))
                ($op)(x::Number, y::Tensor) = ($op)(Constant(x), y)
		($op)(x::Tensor, y::Number) = ($op)(x, Constant(y))
	end
end

@binary_op(Base.:+)
@binary_op(Base.:-)
@binary_op(Base.:*)
@binary_op(Base.:/)
@binary_op(Base.:^)


