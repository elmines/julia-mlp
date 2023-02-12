
export Tensor, Parameter, Operation, Constant

nextId::Int64 = 0
function getNewId()
	global nextId = nextId + 1
	return nextId
end

abstract type Tensor end

struct Parameter <: Tensor
	id::Int64
	trainable::Bool
	Parameter(trainable::Bool) = new(getNewId(), trainable)
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

getId(x::Parameter) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id

