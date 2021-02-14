target = System.get_env("EXLA_TARGET", "host")

defmodule EXLAHelpers do
  @doc """
  Returns the default EXLA client.
  """
  def client(), do: EXLA.Client.fetch!(:default)

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(shapes, fun, opts \\ []) do
    builder = EXLA.Builder.new("test")

    {params, _} =
      Enum.map_reduce(shapes, 0, fn shape, pos ->
        {EXLA.Op.parameter(builder, pos, shape, <<?a + pos>>), pos + 1}
      end)

    fun
    |> apply([builder | params])
    |> EXLA.Builder.build()
    |> EXLA.Computation.compile(client(), shapes, opts)
  end

  @doc """
  Compiles and runs the given function.

  It expects a list of buffers which will be have their shapes
  used for compilation and then given on execution.
  """
  def run(args, opts \\ [], fun) do
    exec = compile(Enum.map(args, & &1.shape), fun)
    EXLA.Executable.run(exec, args, opts)
  end
end

ExUnit.start(
  exclude: [:platform],
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)
