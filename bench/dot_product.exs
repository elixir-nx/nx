cpu = Exla.Client.create_client(platform: :host)
cuda = Exla.Client.create_client(platform: :cuda)

t1 = for _ <- 1..1_000_000, do: 1000
t1_bin = for i <- t1, do: <<i::32-little>>, into: <<>>
t1_shape = Exla.Shape.make_shape(:int32, {1_000_000})
t1_tensor = %Exla.Tensor{data: {:binary, t1_bin}, shape: t1_shape, device: {:beam, 0}}
t1_cpu_ref = Exla.Tensor.to_device(cpu, t1_tensor, {:cpu, 0})

build_execs = fn ->
  builder = Exla.Builder.new("dot")
  x = Exla.Op.parameter(builder, 0, t1_tensor.shape, "x")
  ast = Exla.Op.dot(x, x)
  comp = Exla.Builder.build(ast)
  cpu_exec = Exla.Client.compile(cpu, comp, {t1_tensor.shape})
  gpu_exec = Exla.Client.compile(cuda, comp, {t1_tensor.shape})
  {cpu_exec, gpu_exec}
end

{cpu_exec, gpu_exec} = build_execs.()

elixir_dot = fn a, b ->
  a
  |> Enum.zip(b)
  |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
end

# My GPU is too small and right now our memory management is inefficient so we have to run the GPU benchmarks
# separately
Benchee.run(
  %{
    "elixir dot" => fn -> elixir_dot.(t1, t1) end,
    "xla cpu dot" => fn -> Exla.LocalExecutable.run(cpu_exec, {t1_tensor}) end,
    "xla gpu dot" => fn -> Exla.LocalExecutable.run(gpu_exec, {t1_tensor}) end,
    "xla cpu dot pre-loaded" => fn -> Exla.LocalExecutable.run(cpu_exec, {t1_cpu_ref}) end
    # "xla gpu dot pre-loaded" => fn -> Exla.LocalExecutable.run(gpu_exec, {tensor}) end
  },
  time: 10,
  memory_time: 2
)
