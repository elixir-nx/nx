defmodule EXLAHelpers do
  @doc """
  Returns the default EXLA client.
  """
  def client(), do: EXLA.Client.fetch!(EXLA.Client.default_name())

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(typespecs, opts \\ [], output \\ nil, fun) do
    compile_fn = fn builder ->
      params = EXLA.MLIR.Function.get_arguments(builder)

      fun
      |> apply([builder | params])
      |> then(&EXLA.MLIR.Value.func_return(builder, List.wrap(&1)))

      EXLA.MLIR.Module.compile(
        builder.module,
        client(),
        Enum.map(params, &EXLA.MLIR.Value.get_typespec/1),
        builder.return_typespecs,
        opts
      )
    end

    EXLA.MLIR.Module.new(List.wrap(typespecs), List.wrap(output), compile_fn)
  end

  @doc """
  Compiles and runs the given function.

  It expects a list of buffers which will be have their shapes
  used for compilation and then given on execution.
  """
  def run_one(args, opts \\ [], output \\ nil, fun) do
    exec = compile(Enum.map(args, & &1.typespec), opts, output, fun)
    [result] = EXLA.Executable.run(exec, [args], opts)
    result
  end
end
