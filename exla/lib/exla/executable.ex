defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{Buffer, Shape}

  @enforce_keys [:client, :ref, :output_shape, :num_replicas, :num_partitions]
  defstruct [:client, :ref, :output_shape, :num_replicas, :num_partitions, :async]

  @doc """
  Runs the given executable with arguments.

  ## Options

    *  `:run_id` - a positive integer identifier of this execution.
      One is generated automatically if none is given.

    * `:keep_on_device` - if the data should be kept on the device
      after the computation (defaults to `false`).

    * `:replica` - the replica to run the executable on

  Some options apply to TPU only and therefore are not currently supported:

    * `:launch_id` - the launch id used to coordinate multi-device launches

    * `:rng_seed` - the seed for random numbers

    * `:partition` - the partition within a replica to run the executable on

  """
  def run(%Executable{} = executable, arguments, options \\ []) do
    %{client: client, output_shape: output_shape} = executable
    {data, _} = run(client, executable, arguments, options)
    decompose_output(data, output_shape, client)
  end

  defp run(client, executable, arguments, options) do
    %{ref: exec} = executable

    keep_on_device = Keyword.get(options, :keep_on_device, false)
    keep_on_device_int = if keep_on_device, do: 1, else: 0

    # TODO: Validate replicas against the client
    # TODO: Raise if buffers belong to different clients/ordinals

    inputs =
      Enum.map(arguments, fn
        %Buffer{ref: {ref, _}, data: nil} ->
          ref

        %Buffer{data: data, shape: shape, ref: nil} ->
          {data, shape.ref}
      end)

    data =
      case client.platform do
        :host ->
          EXLA.NIF.run_cpu(
            client.ref,
            exec,
            inputs,
            keep_on_device_int
          )

        _ ->
          EXLA.NIF.run_io(
            client.ref,
            exec,
            inputs,
            keep_on_device_int
          )
      end

    unwrap!(data)
  end

  defp decompose_output(data, shape, client) do
    %Shape{dtype: {:t, shapes}} = shape

    Enum.zip_with(data, shapes, fn
      buf, subshape when is_reference(buf) ->
        Buffer.buffer({buf, client.name}, subshape)

      buf, subshape ->
        Buffer.buffer(buf, subshape)
    end)
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
