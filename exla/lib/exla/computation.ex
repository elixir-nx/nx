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

    * `:num_replicas` - the number of replicas this computation will run on
    * `:use_spmd` - enable single-program multiple data

  Currently those options do not have an effect as they related to running the
  same compiled executabled on multiple replicas.

  Some options apply to TPU only and therefore are not currently supported:

    * `:num_partitions` - the number of partitions this computatio will run on

  """
  def compile(
        computation = %Computation{output_shape: output_shape},
        client = %Client{},
        argument_shapes,
        options \\ []
      ) do
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)

    use_spmd = Keyword.get(options, :use_spmd, false)
    use_spmd_int = if use_spmd, do: 1, else: 0

    unless root_tuple_only?(output_shape) do
      raise ArgumentError,
            "can only compile computations with a tuple at the root (and only at the root), " <>
              "got: #{inspect(output_shape)}"
    end

    # TODO: Validate replicas and partitions against the client

    ref =
      EXLA.NIF.compile(
        client.ref,
        computation.ref,
        Enum.map(argument_shapes, & &1.ref),
        num_replicas,
        num_partitions,
        use_spmd_int
      )
      |> unwrap!()

    %Executable{
      client: client,
      ref: ref,
      output_shape: output_shape,
      num_replicas: num_replicas,
      num_partitions: num_partitions
    }
  end

  defp root_tuple_only?(shape) do
    case shape do
      %{dtype: {:t, inner}} -> Enum.all?(inner, &(not match?({:t, _}, &1.dtype)))
      %{} -> false
    end
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
