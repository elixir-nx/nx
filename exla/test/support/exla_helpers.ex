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
    compile_fn = fn builder ->
      params = EXLA.MLIR.Function.get_arguments(builder)

      fun
      |> apply([builder | params])
      |> then(&EXLA.MLIR.Value.variadic_return(builder, List.wrap(&1)))

      EXLA.MLIR.Module.compile(
        builder.module,
        client(),
        Enum.map(params, &EXLA.MLIR.Value.get_shape/1),
        builder.return_shape,
        opts
      )
    end

    shapes = exla_shape(shapes)
    output = exla_shape(output)
    EXLA.MLIR.Module.new(List.wrap(shapes), List.wrap(output), compile_fn)
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

  defp exla_shape(tensors) when is_list(tensors) do
    Enum.flat_map(tensors, &exla_shape/1)
  end

  defp exla_shape(%{type: :token}) do
    [EXLA.Shape.make_token_shape()]
  end

  defp exla_shape(%{shape: shape, type: type}) do
    [EXLA.Shape.make_shape(type, shape)]
  end

  defp exla_shape(%EXLA.Shape{} = shape) do
    [shape]
  end

  defp exla_shape(%EXLA.MLIR.Value{} = value) do
    [EXLA.MLIR.Value.get_shape(value)]
  end
end
