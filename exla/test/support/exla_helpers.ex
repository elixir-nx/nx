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
    builder =
      if opts[:compiler_mode] != :mlir do
        EXLA.Builder.new("test")
      else
        new_mlir_function("test", List.wrap(shapes), List.wrap(output), false)
      end

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

  defp new_mlir_function(module_and_name, arg_shapes, return_shape, output_tuple?) do
    in_shape = exla_shape(arg_shapes)
    out_shape = exla_shape(return_shape)

    out_shape =
      if output_tuple? do
        [EXLA.Shape.make_tuple_shape(out_shape)]
      else
        out_shape
      end

    EXLA.Builder.new_mlir(module_and_name, in_shape, out_shape)
  end

  defp exla_shape(tensors) when is_list(tensors) do
    Enum.flat_map(tensors, &exla_shape/1)
  end

  defp exla_shape(tensors) when is_tuple(tensors) do
    tensors
    |> Tuple.to_list()
    |> Enum.flat_map(&exla_shape/1)
  end

  defp exla_shape(%Nx.Tensor{type: {:tuple, _size}, data: %{args: args}}) do
    Enum.flat_map(args, &exla_shape/1)
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
