import Base.:+
import Base.:-
import Base.:*
import Base.:/
import Base.:^

nextId::Int64 = 0
function getNewId()
	global nextId = nextId + 1
	return nextId
end

abstract type Tensor end

struct Input <: Tensor
	id::Int64
	Input() = new(getNewId())
end

struct Operation <: Tensor
	id::Int64
	parents::Vector{<:Tensor}
	callback::Function
	Operation(parents::Vector{<:Tensor}, callback::Function) = new(getNewId(), parents, callback)
end

struct Constant <: Tensor
	id::Int64
	value::Number
	Constant(value::Number) = new(getNewId(), value)
end

getId(x::Input) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id

InputDict = Dict{Tensor, Number}

function forward(x::Input, inputs::InputDict)::Number
	return inputs[x]
end

function forward(x::Operation, inputs::InputDict)::Number
	callbackArgs::Vector{Number} = [forward(parent, inputs) for parent in x.parents]
	return (x.callback)(callbackArgs...)
end

function forward(x::Constant, inputs::InputDict)::Number
	return x.value
end

(+)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> a + b)
(-)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> a - b)
(*)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> a * b)
(/)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> a / b)
(^)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> a ^ b)


macro binary_op(op)
	return quote
		($op)(x::Tensor, y::Tensor) = Operation([x, y], (a, b) -> ($op)(a, b))
	end
end
println( @macroexpand @binary_op(*) )
@binary_op(*)


l = Input()
w = Input()
a = l * w
println("rectangle_area = ", forward(a, InputDict(l => 2, w => 6)))

two_tensor = Constant(2)
pi_tensor = Constant(π)
r = Input()
circle_area = pi_tensor * r ^ two_tensor
println("circle_area = ", forward(circle_area, InputDict(r => 1)))
println("circle_area = ", forward(circle_area, InputDict(r => 2)))


