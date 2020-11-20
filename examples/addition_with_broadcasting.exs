client = Exla.Client.create_client()
builder = Exla.Builder.new("broadcasting")

t1 = [1, 2, 3, 4]
t2 = [[5, 6]]

# XLA Handles the Shape so we can just flatten the lists
t1_bin = for i <- t1, do: <<i::32-little>>, into: <<>>
t2_bin = for i <- List.flatten(t2), do: <<i::32-little>>, into: <<>>

t1_shape = Exla.Shape.make_shape(:int32, {4})
t2_shape = Exla.Shape.make_shape(:int32, {1, 2})

t1_tensor = %Exla.Tensor{data: {:binary, t1_bin}, shape: t1_shape, device: {:beam, self()}}
t2_tensor = %Exla.Tensor{data: {:binary, t2_bin}, shape: t2_shape, device: {:beam, self()}}

x = Exla.Op.parameter(builder, 0, t1_shape, "x")
y = Exla.Op.parameter(builder, 1, t2_shape, "y")

# These shapes are incompatible, so if we don't specify a dimension to broadcast along, we get an error!
# Broadcast along the 0th dimension
comp = Exla.Builder.build(Exla.Op.add(x, y, {0}))
exec = Exla.Client.compile(client, comp, {t1_shape, t2_shape})

%Exla.Tensor{data: {:ref, ref}} = Exla.LocalExecutable.run(exec, {t1_tensor, t2_tensor})
literal = Exla.NIF.shaped_buffer_to_literal(client.ref, ref)

IO.puts Exla.NIF.literal_to_string(literal)
