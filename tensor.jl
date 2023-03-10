
export Tensor, Parameter, Operation, Constant, ConcreteTensor
export SizeType

nextId::Int64 = 0
function getNewId()
	global nextId = nextId + 1
	return nextId
end

abstract type Tensor{N} end

SizeType = Tuple{Vararg{<:Integer}}

ConcreteTensor = Union{Array{<:Number}, <:Number}

struct Parameter{N} <: Tensor{N}
	id::Int64
	size::SizeType
	trainable::Bool
	Parameter(size::SizeType, trainable::Bool) = new{length(size)}(getNewId(), size, trainable)
end

struct Operation{N} <: Tensor{N}
	id::Int64
	size::SizeType
	parents::Vector{<:Tensor}
	callback::Function
	Operation(parents::Vector{<:Tensor}, size::SizeType, callback::Function) = new{length(size)}(getNewId(), size, parents, callback)
end

struct Constant{N} <: Tensor{N}
	id::Int64
	value::Union{Number, Array{<:Number, N}}
	Constant(value::Number) = new{0}(getNewId(), value)
	Constant(value::Array{<:Number, M}) where {M} = new{M}(getNewId(), value)
end

function Base.show(io::IO, x::Parameter)
	string_rep = "{Parameter: " *
		"id=$(x.id)" *
		", size=$(x.size)" *
		", trainable=$(x.trainable)" *
		"}"
	show(io, string_rep)
end

function Base.show(io::IO, x::Operation)
	parent_ids = [getId(p) for p in x.parents]
	string_rep = "{Operation: " *
		"id=$(x.id)" *
		", size=$(x.size)" *
		", parents=$(parent_ids)" *
		"}"
	show(io, string_rep)
end

function Base.show(io::IO, x::Constant)
	string_rep = "{Constant: " *
		"id=$(x.id)" *
		", size=$(size(x))" *
		"}"
	show(io, string_rep)
end


getId(x::Parameter) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id

Base.size(x::Parameter) = x.size
Base.size(x::Operation) = x.size
Base.size(x::Constant) = Base.size(x.value)



