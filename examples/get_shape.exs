builder = Exla.Builder.new("builder")

shape = Exla.Shape.make_shape(:float64, [5, 5, 5, 5, 5])

param = Exla.Op.parameter(builder, 0, shape, "x")

IO.inspect Exla.Op.get_shape(param)
