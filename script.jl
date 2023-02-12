include("minesnet.jl")

using .MinesNet

l = Input()
w = Input()
a = l * w
println("rectangle_area = ", forward(a, InputDict(l => 2, w => 6)))

r = Input()
pie = Variable(3.14)
circle_area = pie * r ^ 2
println("circle_area = ", forward(circle_area, InputDict(r => 1)))
println("circle_area = ", forward(circle_area, InputDict(r => 2)))

pie.value = π
println("circle_area = ", forward(circle_area, InputDict(r => 1)))
println("circle_area = ", forward(circle_area, InputDict(r => 2)))

