defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{Buffer, Shape}

  @enforce_keys [:client, :ref, :output_shape, :num_replicas, :num_partitions]
  defstruct [:client, :ref, :output_shape, :num_replicas, :num_partitions, :device_id]

  @doc """
  Runs the given executable with arguments.

  ## Options

    * `:keep_on_device` - if the data should be kept on the device
      after the computation (defaults to `false`).

    * `:device_id` - the device ID to run on

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
            keep_on_device_int,
            executable.device_id
          )

        _ ->
          EXLA.NIF.run_io(
            client.ref,
            exec,
            inputs,
            keep_on_device_int,
            executable.device_id
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
