defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{BinaryBuffer, DeviceBuffer}
  alias EXLA.Typespec

  @enforce_keys [:client, :ref, :output_typespecs, :num_replicas, :num_partitions, :device_id]
  defstruct [
    :client,
    :ref,
    :output_typespecs,
    :num_replicas,
    :num_partitions,
    :device_id,
    :mesh,
    :input_shardings
  ]

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
    %{
      client: client,
      device_id: device_id,
      output_typespecs: output_typespecs,
      ref: ref,
      mesh: mesh
    } =
      executable

    callback_server_pid = Keyword.get(options, :callback_server_pid)
    inputs = prepare_runtime_callback_inputs(inputs, callback_server_pid)

    for data_and_device_id <- run(client, ref, device_id, inputs, options) do
      decompose_output(data_and_device_id, output_typespecs, client, mesh)
    end
  end

  # callback_server_pid_size is generally 8 bytes,
  # but we expose these functions so that we don't
  # hardcode the size in the codebase and are
  # more future-proof.
  def callback_server_pid_size do
    EXLA.NIF.callback_server_pid_size()
  end

  def callback_server_pid_typespec do
    Typespec.tensor({:u, 8}, {callback_server_pid_size()})
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
        device_id: device_id,
        mesh: mesh,
        input_shardings: input_shardings
      })
      when node(ref) == node() do
    serialized_exec =
      ref
      |> EXLA.NIF.serialize_executable()
      |> IO.iodata_to_binary()

    %{
      serialized: serialized_exec,
      output_typespecs: output_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id,
      mesh: mesh,
      input_shardings: input_shardings
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

    ref = EXLA.NIF.deserialize_executable(client.ref, serialized)

    %EXLA.Executable{
      output_typespecs: output_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id,
      mesh: Map.get(data, :mesh),
      input_shardings: Map.get(data, :input_shardings),
      ref: ref,
      client: client
    }
  end

  defp run(client, ref, device_id, inputs, options) do
    callback_server_pid = Keyword.get(options, :callback_server_pid)

    inputs =
      for subinputs <- inputs do
        Enum.map(subinputs, fn
          %DeviceBuffer{ref: ref} ->
            ref

          %BinaryBuffer{data: data, typespec: typespec} ->
            {data, typespec}
        end)
      end

    case client.platform do
      :host -> EXLA.NIF.run_cpu(ref, inputs, device_id, callback_server_pid)
      _ -> EXLA.NIF.run_io(ref, inputs, device_id, callback_server_pid)
    end
  end

  defp decompose_output({data, device_id}, output_typespecs, client, mesh) do
    Enum.zip_with(data, output_typespecs, fn
      buf, logical_typespec when is_reference(buf) ->
        # Query the actual shape from the buffer only in sharded execution
        typespec =
          if mesh != nil do
            EXLA.NIF.get_buffer_typespec(buf)
          else
            logical_typespec
          end

        DeviceBuffer.from_ref(buf, client, device_id, typespec)

      buf, typespec when is_binary(buf) ->
        # Binary buffers use the provided typespec
        BinaryBuffer.from_binary(buf, typespec)
    end)
  end

  defp prepare_runtime_callback_inputs(inputs, nil) do
    callback_server_pid_buffer =
      callback_server_pid_buffer(<<0::size(callback_server_pid_size())-unit(8)>>)

    Enum.map(inputs, fn replica_inputs ->
      [callback_server_pid_buffer | replica_inputs]
    end)
  end

  defp prepare_runtime_callback_inputs(inputs, callback_server_pid) do
    callback_server_pid_buffer =
      callback_server_pid
      |> encode_callback_server_pid!()
      |> callback_server_pid_buffer()

    Enum.map(inputs, fn replica_inputs ->
      [callback_server_pid_buffer | replica_inputs]
    end)
  end

  defp encode_callback_server_pid!(callback_server_pid) do
    callback_server_pid_bin = EXLA.NIF.encode_local_pid(callback_server_pid)
    callback_server_pid_size = callback_server_pid_size()

    case byte_size(callback_server_pid_bin) do
      ^callback_server_pid_size ->
        callback_server_pid_bin

      size ->
        raise ArgumentError,
              "expected encoded callback server pid size to be #{callback_server_pid_size} bytes, got #{size}"
    end
  end

  defp callback_server_pid_buffer(callback_server_pid_bin) do
    BinaryBuffer.from_binary(
      callback_server_pid_bin,
      callback_server_pid_typespec()
    )
  end
end
