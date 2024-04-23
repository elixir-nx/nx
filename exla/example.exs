IO.gets("#{System.pid()} - Press enter to continue")
Nx.default_backend({EXLA.Backend, client: :cuda})
t = Nx.tensor([1, 2, 3])
client = EXLA.Client.fetch!(:cuda)
buffer = t.data.buffer
IO.inspect(t, limit: :infinity)
IO.inspect(t.data, limit: :infinity)

IO.puts("Loading another buffer with a local pointer")
{:ok, {pointer, size}} = EXLA.NIF.get_buffer_device_pointer(client.ref, buffer.ref, :cuda_local) |> IO.inspect(limit: :infinity)
{:ok, new_buffer_ref} = EXLA.NIF.create_buffer_from_device_pointer(client.ref, pointer, :cuda_local, buffer.shape.ref, buffer.device_id)

t2 = put_in(t.data.buffer.ref, new_buffer_ref)

t2 |> IO.inspect(label: "t2")
t2.data |> IO.inspect(label: "t2 data")

IO.inspect(Nx.add(t2, 10))

IO.puts("Loading another buffer with an IPC pointer")
handle = IO.gets("Enter IPC handle bytes: ") |> String.trim()

handle = handle |> String.split([" ", ","], trim: true) |> Enum.map(&String.to_integer/1)

IO.inspect(handle, label: "handle")

{:ok, new_buffer_ref} = EXLA.NIF.create_buffer_from_device_pointer(client.ref, handle, :cuda_ipc, buffer.shape.ref, buffer.device_id)

t3 = put_in(t.data.buffer.ref, new_buffer_ref)

t3 |> IO.inspect(label: "t2")
t3.data |> IO.inspect(label: "t2 data")

IO.inspect(Nx.add(t3, 10))
