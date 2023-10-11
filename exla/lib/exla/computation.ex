defmodule EXLA.Computation do
  @moduledoc """
  Wrapper around XLA's computation.
  """

  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]

  alias EXLA.{Client, Computation, Executable}

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

  def compile(computation = %Computation{}, client = %Client{}, argument_shapes, options) do
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)

    # JAX comments say SPMD can lead to subtle bugs so they only enable
    # when strictly necessary, which is when num_partitions is greater than 1.
    use_spmd = if Keyword.get(options, :use_spmd, true) or num_partitions >= 1, do: 1, else: 0

    device_id =
      if num_replicas > 1 or num_partitions > 1,
        do: -1,
        else: Keyword.get(options, :device_id, client.default_device_id)

    output_shape = assert_output_shape!(computation)

    ref =
      EXLA.NIF.compile(
        client.ref,
        computation.ref,
        Enum.map(argument_shapes, & &1.ref),
        num_replicas,
        num_partitions,
        use_spmd,
        device_id
      )
      |> unwrap!()

    %Executable{
      client: client,
      ref: ref,
      output_shape: output_shape,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id
    }
  end

  def compile(
        %EXLA.MLIR.Function{module: module, return_shape: return_shape},
        client,
        arg_shapes,
        _opts
      ) do
    assert_output_shape!(%{output_shape: return_shape})

    EXLA.MLIR.Module.compile(
      module,
      client,
      arg_shapes,
      return_shape
    )
  end

  defp assert_output_shape!(%{output_shape: output_shape}) do
    if root_tuple_only?(output_shape) do
      output_shape
    else
      raise ArgumentError,
            "can only compile computations with a tuple at the root (and only at the root), " <>
              "got: #{inspect(output_shape)}"
    end
  end

  defp root_tuple_only?(shape) do
    case shape do
      %{dtype: {:tuple, inner}} -> Enum.all?(inner, &(not match?({:tuple, _}, &1.dtype)))
      %{} -> false
    end
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
