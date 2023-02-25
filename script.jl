include("minesnet.jl")

using .MinesNet

l = Parameter((), false)
w = Parameter((), false)

println(l)
println(w)

a = l * w
println(a)
println("rectangle_area = ", forward(a, InputDict(l => 2, w => 6)))
#
#r = Parameter((), false)
#pie = Parameter((), false)
#circle_area = pie * r ^ 2
#println("circle_area = ", forward(circle_area, InputDict(r => 1, pie => 3.14)))
#println("circle_area = ", forward(circle_area, InputDict(r => 2, pie => 3.14)))
#
#println("circle_area = ", forward(circle_area, InputDict(r => 1, pie => π)))
#println("circle_area = ", forward(circle_area, InputDict(r => 2, pie => π)))

