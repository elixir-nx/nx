cpu = Exla.Client.create_client(platform: :host)
cuda = Exla.Client.create_client(platform: :cuda)

# For small lists, Elixir wins big which is expected because of overhead associated with XLA
# The BEAM Crashes on my machine at 100_000_000 elements, XLA handles 100_000_000 and even 1_000_000_000 elements
# with ease AND faster than Elixir with 10_000_000 elements. This isn't even running on a GPU yet...

t1 = for _ <- 1..1_000_000, do: 1000
t1_bin = for i <- t1, do: <<i::32-little>>, into: <<>>
t1_shape = Exla.Shape.make_shape(:int32, {1_000_000})
t1_tensor = %Exla.Tensor{data: {:binary, t1_bin}, shape: t1_shape, device: {:beam, 0}}
t1_cpu_ref = Exla.Tensor.to_device(cpu, t1_tensor, {:cpu, 0})

elixir_dot =
  fn a, b ->
    a
    |> Enum.zip(b)
    |> Enum.reduce(0, fn {x, y}, acc -> acc + (x*y) end)
  end

Benchee.run(%{
  "elixir dot" => fn _ -> elixir_dot.(t1, t1) end,
  "xla cpu dot" => fn {exec, _, _} -> Exla.LocalExecutable.run(exec, {t1_tensor}) end,
  "xla gpu dot" => fn {_, exec, _} -> Exla.LocalExecutable.run(exec, {t1_tensor}) end,
  "xla cpu dot pre-loaded" => fn {exec, _, _} -> Exla.LocalExecutable.run(exec, {t1_cpu_ref}) end, 
  "xla gpu dot pre-loaded" => fn {_, exec, tensor} -> Exla.LocalExecutable.run(exec, {tensor}) end
},
  time: 10,
  memory_time: 2,
  before_each:
  fn _ ->
    t1_gpu_ref = Exla.Tensor.to_device(cuda, t1_tensor, {:cuda, 0})
    builder = Exla.Builder.new("dot")
    x = Exla.Op.parameter(builder, 0, t1_tensor.shape, "x")
    ast = Exla.Op.dot(x, x)
    comp = Exla.Builder.build(ast)
    cpu_exec = Exla.Client.compile(cpu, comp, {t1_tensor.shape})
    gpu_exec = Exla.Client.compile(cuda, comp, {t1_tensor.shape})
    {cpu_exec, gpu_exec, t1_gpu_ref}
  end)
