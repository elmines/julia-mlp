
export Tensor, Parameter, Operation, Constant
export SizeType

nextId::Int64 = 0
function getNewId()
	global nextId = nextId + 1
	return nextId
end

abstract type Tensor end

SizeType = Tuple{Vararg{<:Int}}

struct Parameter <: Tensor
	id::Int64
	size::SizeType
	trainable::Bool
	Parameter(size::SizeType, trainable::Bool) = new(getNewId(), size, trainable)
end

struct Operation <: Tensor
	id::Int64
	size::SizeType
	parents::Vector{<:Tensor}
	callback::Function
	Operation(parents::Vector{<:Tensor}, size::SizeType, callback::Function) = new(getNewId(), size, parents, callback)
end

struct Constant <: Tensor
	id::Int64
	size::SizeType
	value::Number
	Constant(value::Number) = new(getNewId(), Base.size(value), value)
end

getId(x::Parameter) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id

Base.size(x::Parameter) = x.size
Base.size(x::Operation) = x.size
Base.size(x::Constant) = x.size
