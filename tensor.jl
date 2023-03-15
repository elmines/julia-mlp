
export Tensor, Input, Parameter, Operation, Constant
export SizeType
export make_tensor

nextId::Int64 = 0
function getNewId()
	global nextId = nextId + 1
	return nextId
end

abstract type Tensor{N} end

SizeType = Tuple{Vararg{<:Integer}}

struct Input{N} <: Tensor{N}
	id::Int64
	size::SizeType
	Input(size::SizeType) = new{length(size)}(getNewId(), size)
end

struct Parameter{N} <: Tensor{N}
	id::Int64
	value::Union{Number, Array{<:Number, N}}
	Parameter(size::SizeType) = new{length(size)}(getNewId(), Array{Float32}(undef, size))
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

function make_tensor(x)::Tensor
	if isa(x, Tensor)
		return x
	else
		return Constant(x)
	end
end

getId(x::Input) = x.id
getId(x::Parameter) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id

Base.size(x::Input) = x.size
Base.size(x::Parameter) = size(x.value)
Base.size(x::Operation) = x.size
Base.size(x::Constant) = size(x.value)

Base.length(x::Tensor) = prod(size(x))
