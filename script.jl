#!/usr/bin/env julia

include("minesnet.jl")

using .MinesNet

l = Input((3,))
w = Input(())
s = l .+ w
println(s)

input_dict = InputDict(l => [2, 4, 6], w => 3)
println("s = ", forward(s, input_dict))

M = Input((3, 3))
v = [10, 20, 30]
Mv = M .* v
Mv_dict = InputDict(M => [1 2 3; 4 5 6; 7 8 9])
@show Mv_dict
Mv_res = forward(Mv, Mv_dict)
@show Mv_res


expos = Input((3, 3))
exp_ten = [4, 2, 1] .^ expos
exp_dict = InputDict(expos => [1 2 3; 4 5 6; 7 8 9])
exp_res = forward(exp_ten, exp_dict)
@show exp_res

y = Input(())
z = 50 / y
@show forward(z, InputDict(y => 12.5))

Q = Constant([1 2 3; 4 5 6; 7 8 9])
@show forward(3 * Q, InputDict())
@show forward(Q * 3, InputDict())

# Row vector
a = Constant([10 20 30])
@show forward(a * Q, InputDict())

B = Constant([1 0; 1 0; 1 0])
@show forward(Q * B, InputDict())
