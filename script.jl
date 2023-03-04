include("minesnet.jl")

using .MinesNet

compl_tens = Parameter((8, 4, 16, 64, 32), false)
println(size(compl_tens[2]))
println(size(compl_tens[5, 4, 3, 2, 1]))

l = Parameter((3,), false)
w = Parameter((), false)


a = l .* w
println(size(a))

#input_dict = InputDict(l => [2, 4, 6], w => [1, 2, 3])
#println("rectangle_area = ", forward(a, input_dict))

#r = Parameter((), false)
#pie = Parameter((), false)
#circle_area = pie * r ^ 2
#println("circle_area = ", forward(circle_area, InputDict(r => 1, pie => 3.14)))
#println("circle_area = ", forward(circle_area, InputDict(r => 2, pie => 3.14)))
#
#println("circle_area = ", forward(circle_area, InputDict(r => 1, pie => π)))
#println("circle_area = ", forward(circle_area, InputDict(r => 2, pie => π)))

