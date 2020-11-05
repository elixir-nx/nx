Exla.get_or_create_local_client(:host)

# Under the hood, tensors are just blocks of memory with special indexing schemes
# Ideally, we can represent tensors as binaries with a shape and dtype
bin =
  <<0::float, 1::float, 2::float, 3::float, 4::float, 5::float, 6::float, 7::float, 8::float>>
  |> :binary.decode_unsigned()
  |> :binary.encode_unsigned(:little)

dtype = :float64
shape = {3, 3}

# To stay consistent with the rest of the API, we start by creating references to shapes...
# A higher level API should abstract this away
shape_reference = Exla.make_shape(dtype, shape)

# The information above would be in a struct, and we'd get full introspection...
# of course we'd have to transform/truncate based on the shape...

shaped_buffer = Exla.binary_to_shaped_buffer(bin, shape_reference)

literal = Exla.shaped_buffer_to_literal(shaped_buffer)

IO.puts Exla.literal_to_string(literal)

