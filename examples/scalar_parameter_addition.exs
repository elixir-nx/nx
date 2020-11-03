Exla.get_or_create_local_client()

# Need to declare the shape of the inputs to our function
arg1_shape = Exla.make_scalar_shape(:int32)
arg2_shape = Exla.make_scalar_shape(:int32)

# This is the computation graph, x and y are placeholders for any input.
x = Exla.parameter(0, arg1_shape, "x")
y = Exla.parameter(1, arg2_shape, "y")
z = Exla.add(x, y)

# XLA has a Build function which builds from the root XlaOp. We probably want to use that one
# and not worry about any of ther other definitions for Build. Then we'd call Exla.build(z)
# and avoid the warning about z being unused.
computation_graph = Exla.build()

# Literals are the easiest way to represent data for now
# This is a rank 0 literal aka a scalar
a = Exla.create_r0(5)
b = Exla.create_r0(5)

# Now we need to put our data onto the device.
# Transfer to Server is really just a memory allocation which returns handles for our data!
args = {Exla.transfer_to_server(a), Exla.transfer_to_server(b)}

# Now we run it and get the resulting literal
# It's easy to turn a literal into a native type, but for now we just print the result
result = Exla.execute_and_transfer(computation_graph, args)

IO.inspect Exla.literal_to_string(result)