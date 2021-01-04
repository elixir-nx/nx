target = System.get_env("EXLA_TARGET", "host")

defmodule Nx.GradHelpers do
  @doc """
  Checks the gradient of numerical function `func`.

  You must hold the function constant on every other
  variable with a partial application of `func`.
  """
  def check_grads!(func, grad_func, x, eps \\ 1.0e-4) do
    est_grad = finite_differences(func, x, eps)
    comp_grad = grad_func.(x)
    approx_equal?(est_grad, comp_grad, x, eps)
  end

  defp approx_equal?(lhs, rhs, x, eps) do
    [value] = Nx.Util.to_flat_list(Nx.abs(Nx.subtract(lhs, rhs)))

    unless value < eps do
      raise """
      expected

      #{inspect(lhs)}

      to be #{eps} within

      #{inspect(rhs)}

      for input

      #{inspect(x)}
      """
    end
  end

  defp finite_differences(func, x, eps) do
    Nx.divide(
      Nx.subtract(
        func.(Nx.add(x, Nx.divide(eps, 2.0))),
        func.(Nx.subtract(x, Nx.divide(eps, 2.0)))
      ),
      eps
    )
  end
end

defmodule ExlaHelpers do
  @doc """
  Returns the default Exla client.
  """
  def client(), do: Exla.Client.fetch!(:default)

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(shapes, fun, opts \\ []) do
    builder = Exla.Builder.new("test")
    replicas = opts[:num_replicas] || 1

    {params, _} =
      Enum.map_reduce(shapes, 0, fn shape, pos ->
        {Exla.Op.parameter(builder, pos, Exla.Shape.shard(shape, replicas), <<?a + pos>>),
         pos + 1}
      end)

    op = apply(fun, [builder | params])
    comp = Exla.Builder.build(op)
    Exla.Client.compile(client(), comp, shapes, opts)
  end

  @doc """
  Compiles and runs the given function.

  It expects a list of buffers which will be have their shapes
  used for compilation and then given on execution.
  """
  def run(args, opts \\ [], fun) do
    exec = compile(Enum.map(args, & &1.shape), fun)
    Exla.Executable.run(exec, args, opts)
  end

  def run_parallel(args, replicas, opts \\ [], fun) do
    exec = compile(Enum.map(args, & &1.shape), fun, num_replicas: replicas)
    Exla.Executable.run_parallel(exec, args, opts)
  end
end

defmodule Nx.ProcessDevice do
  @behaviour Nx.Device

  def allocate(data, _type, _shape, opts) do
    key = Keyword.fetch!(opts, :key)
    Process.put(key, data)
    {__MODULE__, key}
  end

  def read(key), do: Process.get(key) || raise("deallocated")

  def deallocate(key), do: if(Process.delete(key), do: :ok, else: :already_deallocated)
end

client = ExlaHelpers.client()
multi_device = if client.device_count < 2, do: [:multi_device], else: []

case client.name do
  :host ->
    IO.puts("Testing on host platform with #{client.device_count}")

    if client.device_count < 2 do
      cores = System.schedulers_online()

      IO.puts(
        "To run multi-device tests, set XLA_FLAGS=--xla_force_host_platform_device_count=#{cores}"
      )
    end

  platform ->
    IO.puts("Testing on #{Atom.to_string(platform)} platform with #{client.device_count}")

    if client.device_count < 2 do
      # Treat hardware threads as XLA devices
      cores = System.schedulers_online()

      IO.puts(
        "To run multi-device tests, target host device and set XLA_FLAGS=--xla_force_host_platform_device_count=#{
          cores
        }"
      )
    end
end

ExUnit.start(
  exclude: [:platform] ++ multi_device,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)
