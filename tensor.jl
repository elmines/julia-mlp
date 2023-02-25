
export Tensor, Parameter, Operation, Constant
export SizeType

nextId::Int64 = 0
function getNewId()
	global nextId = nextId + 1
	return nextId
end

abstract type Tensor{N} end

SizeType = Tuple{Vararg{<:Integer}}

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

getId(x::Parameter) = x.id
getId(x::Operation) = x.id
getId(x::Constant) = x.id

Base.size(x::Parameter) = x.size
Base.size(x::Operation) = x.size
Base.size(x::Constant) = Base.size(x.value)

function Base.getindex(x::Tensor{N}, i::Vararg{Int, N}) where {N}
	return Operation([x], ()::Tuple, (raw) -> raw[i...])
end

Base.axes(x::Tensor) = Tuple(Base.OneTo(n) for n in size(x))

Base.BroadcastStyle(::Type{<:Tensor}) = Broadcast.Style{Tensor}()

function Base.similar(bc::Base.Broadcast.Broadcasted, ::Type{ElType}) where {ElType}

	new_size = Tuple(length(ax) for ax in axes(bc))

	return Operation([arg for arg in bc.args], new_size, (raw) -> similar(ElType, axes(raw)))
end
