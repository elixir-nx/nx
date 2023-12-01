defmodule EXLAHelpers do
  @doc """
  Returns the default EXLA client.
  """
  def client(), do: EXLA.Client.fetch!(EXLA.Client.default_name())

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(shapes, opts \\ [], output \\ nil, fun) do
    builder = EXLA.Builder.new("test", shapes, output, opts[:compiler_mode] || :xla)

    params =
      if opts[:compiler_mode] == :mlir do
        EXLA.MLIR.Function.get_arguments(builder)
      else
        {params, _} =
          Enum.map_reduce(shapes, 0, fn shape, pos ->
            {EXLA.Op.parameter(builder, pos, shape, <<?a + pos>>), pos + 1}
          end)

        params
      end

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
  def run_one(args, opts \\ [], output \\ nil, fun) do
    exec = compile(Enum.map(args, & &1.shape), opts, output, fun)
    [result] = EXLA.Executable.run(exec, [args], opts)
    result
  end
end
