
function general_dot(x, y)
	if ndims(x) < 1 || ndims(y) < 1
		return x * y
	end
	@show size(x)
	@show size(y)

	result_shape = (size(x)[begin:end-1]..., size(y)[begin+1:end]...)
	x_view = ndims(x) > 2 ? reshape(x, :, size(x)[end])   : x
	y_view = ndims(y) > 2 ? reshape(y, size(y)[begin], :) : y
	mat_result = x_view * y_view
	result = reshape(mat_result, result_shape)
	return result
end
