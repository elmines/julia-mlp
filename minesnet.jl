module MinesNet

ConcreteTensor{N} = Union{<:Number, Array{<:Number}}

include("tensor.jl")
include("graph.jl")
include("ops/ops.jl")

end
