cpu = Exla.Client.create_client(platform: :host)
cuda = Exla.Client.create_client(platform: :cuda)

t1 = for _ <- 1..1_000_000, do: :rand.uniform()
t1_bin = for i <- t1, do: <<i::float-little>>, into: <<>>
t1_shape = Exla.Shape.make_shape(:float64, {1_000_000})
t1_tensor = %Exla.Tensor{data: {:binary, t1_bin}, shape: t1_shape, device: {:beam, 0}}
t1_cpu_ref = Exla.Tensor.to_device(cpu, t1_tensor, {:cpu, 0})

elixir_softmax =
fn a ->
  sum = Enum.reduce(a, 0, &(:math.exp(&1) + &2))
  exps =
    a
    |> Enum.map(&(:math.exp(&1) / sum))

  exps
end

# My GPU is too small and right now our memory management is inefficient so we have to run the GPU benchmarks
# separately
Benchee.run(%{
  "elixir softmax" => fn _ -> elixir_softmax.(t1) end,
  "xla cpu softmax" => fn {exec, _} -> Exla.LocalExecutable.run(exec, {t1_tensor}) end,
  "xla cpu sotfmax ref" => fn {exec, _} -> Exla.LocalExecutable.run(exec, {t1_cpu_ref}) end,
  # "xla gpu softmax ref" => fn {exec, t1_gpu_ref} -> Exla.LocalExecutable.run(exec, {t1_gpu_ref}) end
  "xla gpu softmax" => fn {_, exec} -> Exla.LocalExecutable.run(exec, {t1_tensor}) end
  },
  time: 10,
  memory_time: 2,
  before_each: fn _ ->
    # t1_gpu_ref = Exla.Tensor.to_device(cuda, t1_tensor, {:cuda, 0})
    builder = Exla.Builder.new("softmax")
    # We need a sub-builder of builder because we need to create a sub-computation
    # My understanding is that sub-builders are used to generate computations
    # within the same scope of a larger computation.
    sub_builder = Exla.Builder.new(builder, "softmax_child")
    # The sub-builder takes it's own parameters for the reduction computation
    reduction_shape = Exla.Shape.make_shape(:float64, {})
    a = Exla.Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Exla.Op.parameter(sub_builder, 1, reduction_shape, "b")
    reduction_ast = Exla.Op.add(a, b)
    # Now we build the reduction computation to use in the overall computation
    reduction = Exla.Builder.build(reduction_ast)
    # The overall computation takes a tensor parameter
    shape = Exla.Shape.make_shape(:float64, {1_000_000})
    x = Exla.Op.parameter(builder, 0, shape, "x")
    # Element-wise unary exponential
    exp_x = Exla.Op.exp(x)
    # Initial value is constant 0
    init_value = Exla.Op.zero(builder, :float64)
    # We apply the reduction along the first axis
    divisor = Exla.Op.reduce(x, init_value, reduction, {0})
    # The divisor is a scalar, which is automatically broadcasted
    result = Exla.Op.div(exp_x, divisor)

    comp = Exla.Builder.build(result)
    cpu_exec = Exla.Client.compile(cpu, comp, {shape})
    gpu_exec = Exla.Client.compile(cuda, comp, {shape})

    {cpu_exec, gpu_exec}
  end
)
