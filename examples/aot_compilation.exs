# We can build the AST in Elixir, but there are some extra
# things the codegen expects (like a tuple output), so we need
# to go through everything here:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/xla_compiler.cc#L1284
# to make sure we're getting everything right
builder = Exla.Builder.new("add_2")

shape = Exla.Shape.make_shape({:s, 32}, {2, 1000})

x = Exla.Op.parameter(builder, 0, shape, "x")
y = Exla.Op.parameter(builder, 1, shape, "y")

ast = Exla.Op.tuple(builder, [Exla.Op.add(x, y)])

# The only thing we need is the computation for a function at the end
comp = Exla.Builder.build(ast)

:ok = Exla.NIF.compile_aot(comp.ref)