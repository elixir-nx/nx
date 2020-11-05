# This program computes z = x + y + C with the parameters x = 1, y = 1

# Initialize the local client, this just runs on CPU
Exla.get_or_create_local_client()

# We'll start by declaring the shapes of our parameters
arg1_shape = Exla.make_scalar_shape(:int32)
arg2_shape = Exla.make_scalar_shape(:int32)

# Now we declare the parameters and a constant
x = Exla.parameter(0, arg1_shape, "x")
y = Exla.parameter(1, arg2_shape, "y")
c = Exla.constant_r0(1)

# Now we get the result, this is the root op
z = Exla.add(Exla.add(x, y), c)

# Build computation from root
comp = Exla.build(z)

# Compile to a local executable, we must include argument shapes so it can be built in to the executable
exec = Exla.compile(comp, {arg1_shape, arg2_shape}, %Exla.ExecutableBuildOptions{})

# Now we declare two literals which we can pass to the executable when we run
a = Exla.create_r0(1)
b = Exla.create_r0(1)

# Place both of these on the XLA device
# You need to specify a device ordinal, which is basically the id of the device
# You also can specify a memory allocator to use, for now we leave it nil which
# uses the default allocator
a_buffer = Exla.literal_to_shaped_buffer(a, 0, nil)
b_buffer = Exla.literal_to_shaped_buffer(b, 0, nil)

# Now we run the computation with our arguments
# For some reason compile returns a list of executables
result = Exla.run(exec, {a_buffer, b_buffer}, %Exla.ExecutableRunOptions{})

# Transfer this back to a literal
result_literal = Exla.shaped_buffer_to_literal(result)

# Inspect the result
IO.inspect Exla.literal_to_string(result_literal)
