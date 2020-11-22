cpu = Exla.Client.create_client(platform: :host)
# cuda = Exla.Client.create_client(platform: :cuda)

size = 1_000_000
t1 = for _ <- 1..size, do: :rand.uniform()
t1_nx = Nx.tensor(t1)
t1_shape = Exla.Shape.make_shape(:float64, {size})
t1_tensor = %Exla.Tensor{data: {:binary, Nx.to_bitstring(t1_nx)}, shape: t1_shape, device: {:beam, 0}}
t1_cpu_ref = Exla.Tensor.to_device(cpu, t1_tensor, {:cpu, 0})

build_execs =
  fn ->
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
    shape = Exla.Shape.make_shape(:float64, {size})
    x = Exla.Op.parameter(builder, 0, shape, "x")
    # Element-wise unary exponential
    exp_x = Exla.Op.exp(x)
    # Initial value is constant 0
    init_value = Exla.Op.zero(builder, :float64)
    # We apply the reduction along the first axis
    divisor = Exla.Op.reduce(exp_x, init_value, reduction, {0})
    # The divisor is a scalar, which is automatically broadcasted
    result = Exla.Op.div(exp_x, divisor)

    comp = Exla.Builder.build(result)
    cpu_exec = Exla.Client.compile(cpu, comp, {shape})
    gpu_exec = nil # Exla.Client.compile(cuda, comp, {shape})

    {cpu_exec, gpu_exec}
  end

{cpu_exec, gpu_exec} = build_execs.()


defmodule Softmax do
  import Nx.Defn

  defn elixir(n) do
    Nx.exp(n) / Nx.sum(Nx.exp(n))
  end
end

# IO.inspect Softmax.elixir(t1_nx)
# IO.inspect Exla.LocalExecutable.run(cpu_exec, {t1_tensor})

# My GPU is too small and right now our memory management is inefficient so we have to run the GPU benchmarks
# separately
Benchee.run(%{
  "elixir softmax" => fn -> Softmax.elixir(t1_nx) end,
  "xla cpu softmax" => fn -> Exla.LocalExecutable.run(cpu_exec, {t1_tensor}) end,
  # "xla gpu softmax ref" => fn {exec, t1_gpu_ref} -> Exla.LocalExecutable.run(exec, {t1_gpu_ref}) end
  # "xla gpu softmax" => fn -> Exla.LocalExecutable.run(gpu_exec, {t1_tensor}) end
  },
  time: 10,
  memory_time: 2
)
