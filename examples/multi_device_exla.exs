client = Exla.Client.fetch!(:default)

# Start by taking a look at the number of devices available to the client:
IO.puts "Available Devices: #{client.device_count}\n"

# And the default device ordinal
IO.puts "Default Device Ordinal: #{client.default_device_ordinal}"

# Build a simple computation
builder = Exla.Builder.new("add_2")
shape = Exla.Shape.make_shape({:s, 32}, {2, 1000})
x = Exla.Op.parameter(builder, 0, shape, "x")
y = Exla.Op.parameter(builder, 1, shape, "y")
ast = Exla.Op.add(x, y)
comp = Exla.Builder.build(ast)

# Now compile for the default device:
exec_default = Exla.Client.compile(client, comp, [shape, shape])

# And run it on some buffers:
data = for i <- 1..2000, into: <<>>, do: <<i::32-native>>

b1 = Exla.Buffer.buffer(data, shape)
b2 = Exla.Buffer.buffer(data, shape)

IO.inspect Exla.Executable.run(exec_default, [b1, b2])

# This will run on the second device
exec_second = Exla.Client.compile(client, comp, [shape, shape], device_ordinal: 1)

IO.inspect Exla.Executable.run(exec_second, [b1, b2])

# If we try to do it on an ordinal that doesn't exist, it will throw:
try do
  Exla.Client.compile(client, comp, [shape, shape], device_ordinal: 5)
rescue
  e in ArgumentError -> IO.inspect e
end