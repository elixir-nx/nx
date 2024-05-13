defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{BinaryBuffer, DeviceBuffer}

  @enforce_keys [:client, :ref, :output_typespecs, :num_replicas, :num_partitions, :device_id]
  defstruct [
    :client,
    :ref,
    :output_typespecs,
    :num_replicas,
    :num_partitions,
    :device_id,
    runtime: :xla
  ]

  @doc """
  Runs the given executable with a list of lists as inputs and the given options.
  """
  def run(%Executable{} = executable, [subinputs | _] = inputs, options \\ [])
      when is_list(subinputs) do
    %{
      runtime: runtime,
      client: client,
      device_id: device_id,
      output_typespecs: output_typespecs,
      ref: ref
    } =
      executable

    for data_and_device_id <- run(runtime, client, ref, device_id, inputs, options) do
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

  defp run(:xla, client, ref, device_id, inputs, _options) do
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

  defp run(:iree, _client, ref, device_id, inputs, _options) do
    dbg()

    inputs =
      for subinputs <- inputs do
        Enum.map(subinputs, fn
          %BinaryBuffer{data: data, typespec: typespec} ->
            if typespec.type in [f: 64, c: 128, s: 64, u: 64] do
              {t, w} = typespec.type
              w2 = div(w, 2)
              target_type = {t, w2}

              data =
                Nx.with_default_backend(Nx.BinaryBackend, fn ->
                  data
                  |> Nx.from_binary(typespec.type)
                  |> Nx.as_type(target_type)
                  |> Nx.to_binary()
                end)

              data = <<data::bitstring, 0::size(w2)>>

              {data, EXLA.Typespec.nif_encode(typespec)}
            else
              {data, EXLA.Typespec.nif_encode(typespec)}
            end
        end)
      end

    ref
    |> EXLA.MLIR.IREE.run(List.flatten(inputs))
    |> unwrap!()
    |> then(&[{&1, device_id}])
  end

  defp decompose_output({data, device_id}, output_typespecs, client) do
    Enum.zip_with(data, output_typespecs, fn
      {type, buf}, target_typespec when is_binary(buf) and is_list(type) ->
        source_typespec = EXLA.Typespec.nif_decode({type, target_typespec.shape})

        if source_typespec == target_typespec do
          BinaryBuffer.from_binary(buf, target_typespec)
        else
          Nx.with_default_backend(Nx.BinaryBackend, fn ->
            buf
            |> Nx.from_binary(source_typespec.type)
            |> Nx.as_type(target_typespec.type)
            |> Nx.to_binary()
            |> BinaryBuffer.from_binary(target_typespec)
          end)
        end

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
