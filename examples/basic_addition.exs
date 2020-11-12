# This program computes z = x + y + C with the parameters x = 1, y = 1

# Initialize the local client, this just runs with default options on CPU
client = Exla.Client.create_client()
builder = Exla.Builder.new("addition")

# We'll start by declaring the shapes of our parameters
arg1_shape = Exla.Shape.make_shape(:int32, {})
arg2_shape = Exla.Shape.make_shape(:int32, {})

# Now we declare the parameters and a constant
x = Exla.Op.parameter(builder, 0, arg1_shape, "x")
y = Exla.Op.parameter(builder, 1, arg2_shape, "y")
c = Exla.Op.constant(builder, 1)

# Now we get the result, this is the root op
z = Exla.Op.add(Exla.Op.add(x, y), c)

# Build computation from root
comp = Exla.Builder.build(z)

# Compile to a local executable, we must include argument shapes so it can be built in to the executable
exec = Exla.Client.compile(client, comp, {arg1_shape, arg2_shape})

# Now we declare two literals which we can pass to the executable when we run
t1 = Exla.Tensor.scalar(1, :int32)
t2 = Exla.Tensor.scalar(1, :int32)

result = Exla.LocalExecutable.run(client, exec, {t1, t2})

IO.inspect result
