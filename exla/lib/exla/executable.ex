defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{BinaryBuffer, Buffer, Shape}

  @enforce_keys [:client, :ref, :output_shape, :num_replicas, :num_partitions, :device_id]
  defstruct [:client, :ref, :output_shape, :num_replicas, :num_partitions, :device_id]

  @doc """
  Runs the given executable with arguments.

  ## Options

    * `:keep_on_device` - if the data should be kept on the device
      after the computation (defaults to `false`).

  """
  def run(%Executable{} = executable, arguments, options \\ []) do
    %{client: client, device_id: device_id, output_shape: output_shape, ref: ref} = executable

    device_id =
      cond do
        opt_device_id = options[:device_id] ->
          opt_device_id

        device_id >= 0 ->
          device_id

        true ->
          raise ArgumentError, ":device_id is expected on run for single-program multiple-data"
      end

    {data, _} = run(client, ref, device_id, arguments, options)
    decompose_output(data, output_shape, client, device_id)
  end

  defp run(client, ref, device_id, arguments, options) do
    keep_on_device = Keyword.get(options, :keep_on_device, false)
    keep_on_device_int = if keep_on_device, do: 1, else: 0

    inputs =
      Enum.map(arguments, fn
        %Buffer{ref: ref} -> ref
        %BinaryBuffer{data: data, shape: shape} -> {data, shape.ref}
      end)

    data =
      case client.platform do
        :host -> EXLA.NIF.run_cpu(client.ref, ref, inputs, keep_on_device_int, device_id)
        _ -> EXLA.NIF.run_io(client.ref, ref, inputs, keep_on_device_int, device_id)
      end

    unwrap!(data)
  end

  defp decompose_output(data, shape, client, device_id) do
    %Shape{dtype: {:tuple, shapes}} = shape

    Enum.zip_with(data, shapes, fn
      buf, subshape when is_reference(buf) ->
        Buffer.from_ref(buf, client, device_id, subshape)

      buf, subshape when is_binary(buf) ->
        BinaryBuffer.from_binary(buf, subshape)
    end)
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
