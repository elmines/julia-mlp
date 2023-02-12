include("minesnet.jl")

using .MinesNet

l = Input()
w = Input()
a = l * w
println("rectangle_area = ", forward(a, InputDict(l => 2, w => 6)))

r = Input()
circle_area = π * r ^ 2
println("circle_area = ", forward(circle_area, InputDict(r => 1)))
println("circle_area = ", forward(circle_area, InputDict(r => 2)))


