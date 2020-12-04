target = System.get_env("EXLA_TARGET", "host")

ExUnit.start(
  exclude: :platform,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)

defmodule ExlaHelpers do
  @doc """
  Returns the default Exla client.
  """
  def client(), do: Exla.Client.fetch!(:default)

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(shapes, fun) do
    builder = Exla.Builder.new("test")

    {params, _} =
      Enum.map_reduce(shapes, 0, fn shape, pos ->
        {Exla.Op.parameter(builder, pos, shape, <<?a + pos>>), pos + 1}
      end)

    op = apply(fun, [builder | params])
    comp = Exla.Builder.build(op)
    Exla.Client.compile(client(), comp, shapes)
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
end

defmodule Nx.ProcessDevice do
  @behaviour Nx.Device

  def allocate(data, _type, _shape, opts) do
    key = Keyword.fetch!(opts, :key)
    Process.put(key, data)
    {__MODULE__, key}
  end

  def read(key), do: Process.get(key) || raise "deallocated"

  def deallocate(key), do: if(Process.delete(key), do: :ok, else: :already_deallocated)
end