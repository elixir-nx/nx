defmodule EXLA.Computation do
  @moduledoc """
  Wrapper around XLA's computation.
  """

  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]

  @doc """
  Compiles a computation into an executable.

  ## Options

    * `:device_id` - the device id to compile to and run the executable on.
      Defaults to the `:default_device_id` on the client. If `:num_replicas`
      or `:num_partitions` are given, this option is ignored and the device
      id is set to `-1`.

    * `:num_replicas` - the number of replicas this computation will run on.
      It defaults to 1 but you can set it if you want to enable single-program
      multiple data.

    * `:use_spmd` - enables Single-Program Multiple-Data partioning.
      This is set to true if `:num_partitions` is more than one, otherwise is `false`.

  Currently those options do not have an effect as they related to running the
  same compiled executable on multiple replicas.

  Some options apply to TPU only:

    * `:num_partitions` - the number of partitions this computation will run on.

  """
  def compile(computation, client, argument_shapes, options \\ [])

  def compile(
        %EXLA.MLIR.Function{module: module, return_shape: return_shape},
        client,
        arg_shapes,
        opts
      ) do
    EXLA.MLIR.Module.compile(
      module,
      client,
      arg_shapes,
      return_shape,
      opts
    )
  end
end
