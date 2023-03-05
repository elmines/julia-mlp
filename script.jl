#!/usr/bin/env julia

include("minesnet.jl")

using .MinesNet
using Random:MersenneTwister

l = Parameter((3,), false)
w = Parameter((), false)
s = l .+ w
println(s)

input_dict = InputDict(l => [2, 4, 6], w => 3)
println("s = ", forward(s, input_dict))

M = Parameter((3, 3), false)
v = [10, 20, 30]
Mv = M .* v
Mv_dict = InputDict(M => [1 2 3; 4 5 6; 7 8 9])
@show Mv_dict
Mv_res = forward(Mv, Mv_dict)
@show Mv_res


expos = Parameter((3, 3), false)
exp_ten = [4, 2, 1] .^ expos
exp_dict = InputDict(expos => [1 2 3; 4 5 6; 7 8 9])
exp_res = forward(exp_ten, exp_dict)
@show exp_res
