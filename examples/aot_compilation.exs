builder = Exla.Builder.new("add_2")

shape = Exla.Shape.make_shape({:s, 32}, {2, 1000})

x = Exla.Op.parameter(builder, 0, shape, "x")
y = Exla.Op.parameter(builder, 1, shape, "y")

ast = Exla.Op.tuple(builder, [Exla.Op.add(x, y)])

comp = Exla.Builder.build(ast)

args = [%{id: 0, name: "x", dims: {2, 1000}}, %{id: 1, name: "y", dims: {2, 1000}}]

Exla.Aot.Compile.compile([comp], [{:my_function, 2, args, 2000}])