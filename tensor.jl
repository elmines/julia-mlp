
export Tensor, Input, Parameter, Operation, Constant
export SizeType
export make_tensor

import Distributions:Normal

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
	name::Union{Missing, String}
	Input(size::SizeType, name::Union{Missing, String} = missing) = new{length(size)}(getNewId(), size, name)
end

struct Parameter{N} <: Tensor{N}
	id::Int64
	value::Union{Number, Array{<:Number, N}}
	name::Union{Missing, String}
	Parameter(initializer::Array{<:Number, N}, name::Union{String, Missing} = missing) where {N} = new{N}(getNewId(), initializer, name)
end

function Parameter(size::SizeType, name::Union{String, Missing} = missing)
	return Parameter(rand(Normal(0, 1), size), name)
end


struct Operation{N} <: Tensor{N}
	id::Int64
	size::SizeType
	parents::Vector{<:Tensor}
	callback::Function
	grad_callbacks::Vector{Function}
	name::Union{String, Missing}
	Operation(parents::Vector{<:Tensor}, size::SizeType, callback::Function, grad_callbacks::Vector{Function}, name::Union{String, Missing} = missing) = new{length(size)}(getNewId(), size, parents, callback, grad_callbacks, name)
end

struct Constant{N} <: Tensor{N}
	id::Int64
	value::Union{Number, Array{<:Number, N}}
	name::Union{String, Missing}
	Constant(value::Number) = new{0}(getNewId(), value)
	Constant(value::Array{<:Number, M}, name::Union{String, Missing} = missing) where {M} = new{M}(getNewId(), value, name)
end

function Base.show(io::IO, x::Input)
	string_rep = "{Input: " * 
		"name=$(name(x))" *
		", size=$(size(x))" *
		"}"
	show(io, string_rep)
end

function Base.show(io::IO, x::Parameter)
	string_rep = "{Parameter: " *
		"name=$(name(x))" *
		", size=$(size(x))" *
		"}"
	show(io, string_rep)
end

function Base.show(io::IO, x::Operation)
	parent_names = join([name(p) for p in x.parents], ",")
	string_rep = "{Operation: " *
		"name=$(name(x))" *
		", size=$(x.size)" *
		", parents=[$(parent_names)]" *
		"}"
	show(io, string_rep)
end

function Base.show(io::IO, x::Constant)
	string_rep = "{Constant: " *
		"name=$(name(x))" *
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

name(x::Tensor) = ismissing(x.name) ? string(x.id) : x.name

Base.length(x::Tensor) = prod(size(x))
