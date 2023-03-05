
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

Base.length(x::Tensor) = prod(size(x))

Base.ndims(x::Tensor) = length(size(x))

Base.ndims(::Type{<:Tensor{N}}) where N = N

function Base.getindex(x::Tensor{N}, i::Vararg{Int, N}) where {N}
	if N < 1
		throw(DimensionMismatch("Tried to index scalar tensor"))
	end
	return Operation([x], (), (raw) -> raw[i...])
end

function Base.getindex(x::Tensor{N}, i::Int) where {N}
	if N < 1
		throw(DimensionMismatch("Tried to index scalar tensor"))
	end
	return Operation([x], size(x)[2:end], (raw) -> raw[i])
end

function Base.getindex(x::Tensor{N}, i::CartesianIndex{N}) where {N}
	return x[Tuple(i)...]
end

Base.axes(x::Tensor) = Tuple(Base.OneTo(n) for n in size(x))

function Base.iterate(x::Tensor)
	#println("Iterating on " * string(x))
	if size(x) == ()
		#println("\tReturning the raw tensor")
		return x
	end
	return (x[1], 2)
end

function Base.iterate(x::Tensor, state)
	#println("Iterating on " * string(x) * " with state " * string(state))
	size_x = size(x)
	if size_x == () || state > size_x[1]
		#println("\tReturning nothing")
		return nothing
	end
	return (x[state], state + 1)
end



# Broadcasting
struct TensorStyle <: Base.BroadcastStyle end

function Base.BroadcastStyle(::Type{<:Tensor{N}}) where N
	println("Called unary")
	return TensorStyle()
end

function Broadcast.broadcastable(x::Tensor)
	println("Broadcasting $x")
	return x
end

