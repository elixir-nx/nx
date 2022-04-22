defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{BinaryBuffer, DeviceBuffer, Shape}

  @enforce_keys [:client, :ref, :output_shape, :num_replicas, :num_partitions, :device_id]
  defstruct [:client, :ref, :output_shape, :num_replicas, :num_partitions, :device_id]

  @doc """
  Runs the given executable with a list of lists as inputs and the given options.
  """
  def run(%Executable{} = executable, [subinputs | _] = inputs, options \\ [])
      when is_list(subinputs) do
    %{client: client, device_id: device_id, output_shape: output_shape, ref: ref} = executable

    for data_and_device_id <- run(client, ref, device_id, inputs, options) do
      decompose_output(data_and_device_id, output_shape, client)
    end
  end

  defp run(client, ref, device_id, inputs, _options) do
    inputs =
      for subinputs <- inputs do
        Enum.map(subinputs, fn
          %DeviceBuffer{ref: ref} -> ref
          %BinaryBuffer{data: data, shape: shape} -> {data, shape.ref}
        end)
      end

    data =
      case client.platform do
        :host -> EXLA.NIF.run_cpu(client.ref, ref, inputs, device_id)
        _ -> EXLA.NIF.run_io(client.ref, ref, inputs, device_id)
      end

    unwrap!(data)
  end

  defp decompose_output({data, device_id}, shape, client) do
    %Shape{dtype: {:tuple, shapes}} = shape

    Enum.zip_with(data, shapes, fn
      buf, subshape when is_reference(buf) ->
        DeviceBuffer.from_ref(buf, client, device_id, subshape)

      buf, subshape when is_binary(buf) ->
        BinaryBuffer.from_binary(buf, subshape)
    end)
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
