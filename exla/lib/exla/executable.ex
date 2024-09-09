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

  Works across nodes.
  """
  def run(executable, inputs, options \\ [])

  def run(%Executable{ref: ref, client: client} = executable, inputs, options)
      when node(ref) != node() do
    client
    |> load(dump(executable))
    |> run(inputs, options)
  end

  def run(%Executable{} = executable, [subinputs | _] = inputs, options)
      when is_list(subinputs) do
    %{client: client, device_id: device_id, output_typespecs: output_typespecs, ref: ref} =
      executable

    for data_and_device_id <- run(client, ref, device_id, inputs, options) do
      decompose_output(data_and_device_id, output_typespecs, client)
    end
  end

  @doc """
  Dumps the executable to a data structure that can be serialized
  with `term_to_binary`.

  Works across nodes.
  """
  # If you change this function, you must bump the version in EXLA.Defn.Disk.
  def dump(%Executable{
        ref: ref,
        output_typespecs: output_typespecs,
        num_replicas: num_replicas,
        num_partitions: num_partitions,
        device_id: device_id
      })
      when node(ref) == node() do
    serialized_exec =
      ref
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
  end

  def dump(%Executable{ref: ref} = executable) do
    :erpc.call(node(ref), __MODULE__, :dump, [executable])
  end

  @doc """
  Loads a previously dumped executable.
  """
  def load(client, data) do
    %{
      serialized: serialized,
      output_typespecs: output_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id
    } = data

    ref =
      serialized
      |> then(&EXLA.NIF.deserialize_executable(client.ref, &1))
      |> unwrap!()

    %EXLA.Executable{
      output_typespecs: output_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id,
      ref: ref,
      client: client
    }
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
