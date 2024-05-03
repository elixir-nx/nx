defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{BinaryBuffer, DeviceBuffer}

  @enforce_keys [:client, :ref, :output_typespecs, :num_replicas, :num_partitions, :device_id]
  defstruct [:client, :ref, :output_typespecs, :num_replicas, :num_partitions, :device_id]

  @doc """
  Runs the given executable with a list of lists as inputs and the given options.
  """
  def run(%Executable{} = executable, [subinputs | _] = inputs, options \\ [])
      when is_list(subinputs) do
    %{client: client, device_id: device_id, output_typespecs: output_typespecs, ref: ref} =
      executable

    for data_and_device_id <- run(client, ref, device_id, inputs, options) do
      decompose_output(data_and_device_id, output_typespecs, client)
    end
  end

  def serialize(%Executable{
        ref: executable,
        output_typespecs: output_typespecs,
        num_replicas: num_replicas,
        num_partitions: num_partitions,
        device_id: device_id
      }) do
    serialized_exec =
      executable
      |> EXLA.NIF.serialize_executable()
      |> unwrap!()
      |> IO.iodata_to_binary()

    %{
      serialized: serialized_exec,
      output_typespecs: output_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id
    }
    |> :erlang.term_to_binary()
  end

  def deserialize(client, binary) do
    case :erlang.binary_to_term(binary) do
      %{serialized: serialized_exec} = exec_data ->
        ref =
          serialized_exec
          |> then(&EXLA.NIF.deserialize_executable(client.ref, &1))
          |> unwrap!()

        exec_data
        |> Map.put(:ref, ref)
        |> Map.put(:client, client)
        |> then(&struct(__MODULE__, &1))

      _other ->
        raise "invalid serialized executable"
    end
  end

  defp run(client, ref, device_id, inputs, _options) do
    inputs =
      for subinputs <- inputs do
        Enum.map(subinputs, fn
          %DeviceBuffer{ref: ref} ->
            ref

          %BinaryBuffer{data: data, typespec: typespec} ->
            {data, EXLA.Typespec.nif_encode(typespec)}
        end)
      end

    data =
      case client.platform do
        :host -> EXLA.NIF.run_cpu(client.ref, ref, inputs, device_id)
        _ -> EXLA.NIF.run_io(client.ref, ref, inputs, device_id)
      end

    unwrap!(data)
  end

  defp decompose_output({data, device_id}, output_typespecs, client) do
    Enum.zip_with(data, output_typespecs, fn
      buf, typespec when is_reference(buf) ->
        DeviceBuffer.from_ref(buf, client, device_id, typespec)

      buf, typespec when is_binary(buf) ->
        BinaryBuffer.from_binary(buf, typespec)
    end)
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
