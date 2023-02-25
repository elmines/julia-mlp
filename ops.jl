macro simple_binary_op(op)
	return quote
		function ($op)(x::Tensor, y::Tensor)
			if size(x) != size(y)
				throw(DimensionMismatch(string(size(x)) * " incompatible with " * string(size(y))))
			end

			return Operation([x, y], size(x), (a, b) -> ($op)(a, b))
		end

                ($op)(x::Number, y::Tensor) = ($op)(Constant(x), y)
		($op)(x::Tensor, y::Number) = ($op)(x, Constant(y))
                ($op)(x::Array, y::Tensor) = ($op)(Constant(x), y)
		($op)(x::Tensor, y::Array) = ($op)(x, Constant(y))

	end
end

@simple_binary_op(Base.:+)
@simple_binary_op(Base.:-)
@simple_binary_op(Base.:/)
@simple_binary_op(Base.:^)
@simple_binary_op(Base.:*)
