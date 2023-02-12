
export Tensor, Input, Operation, Constant, Variable

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

mutable struct Variable <: Tensor
	id::Int64
	value::Number
	Variable(value::Number) = new(getNewId(), value)
end

getId(x::Input) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id
getId(x::Variable) = x.id

