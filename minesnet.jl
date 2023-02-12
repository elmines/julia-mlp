module MinesNet

export Tensor, Input, Operation, Constant
export InputDict, forward

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

end
