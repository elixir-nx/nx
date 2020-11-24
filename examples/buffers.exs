client = Exla.Client.create_client(platform: :cuda)

size = 1_000_000
t = for _ <- 1..size, do: 1000

bin = for i <- t, do: <<i::float-native>>, into: <<>>
shape = Exla.Shape.make_shape(:float64, {size})


for i <- 1..1000 do
  IO.inspect i
  for _ <- 1..50 do
    {:ok, ref} = Exla.NIF.binary_to_shaped_buffer(client.ref, bin, shape.ref, 0)
    Exla.NIF.shaped_buffer_to_binary(client.ref, ref)
  end
  :erlang.garbage_collect(self())
end