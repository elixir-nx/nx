defmodule EXLA.Defn.Stream do
  @moduledoc false

  keys =
    [:lock, :outfeed, :pid, :ref, :send, :recv, :send_shape] ++
      [:recv_shapes, :done, :client, :device_id, :keep_on_device]

  @derive {Inspect, only: [:pid, :client, :device_id, :keep_on_device, :send, :recv]}
  @enforce_keys keys
  defstruct keys

  def run(
        executable,
        lock,
        task,
        outfeed,
        send,
        send_shape,
        recv,
        recv_shapes,
        done,
        keep_on_device?
      ) do
    %{client: client, device_id: device_id} = executable
    %{pid: task_pid, ref: task_ref} = task

    # With the task and outfeed in place, we now relock the client/device_id.
    # If the current process shuts down, we send an infeed to stop the loop.
    ^lock =
      EXLA.Defn.Lock.relock(
        lock,
        fn -> send(task_pid, lock) end,
        fn -> halt_stream(client, device_id, outfeed) end
      )

    %EXLA.Defn.Stream{
      pid: self(),
      ref: task_ref,
      outfeed: outfeed,
      lock: lock,
      send: send,
      send_shape: send_shape,
      recv: recv,
      recv_shapes: recv_shapes,
      client: client,
      device_id: device_id,
      done: done,
      keep_on_device: keep_on_device?
    }
  end

  # It is time to halt the stream, we do it by sending 0 for the loop infeed.
  # Then we wait for the outfeed process to read all.
  defp halt_stream(client, device_id, outfeed) do
    pred = EXLA.Shape.make_shape({:pred, 8}, {})
    :ok = EXLA.Client.to_infeed(client, device_id, [{<<0::8-native>>, pred}])
    {:lock, outfeed, fn -> :unlocked end}
  end

  defimpl Nx.Stream do
    def send(%{client: client, device_id: device_id, send: send, send_shape: send_shape}, data) do
      unless Nx.compatible?(send, data) do
        raise ArgumentError, """
        Nx stream expected a tensor of type, shape, and names on send:

        #{inspect(send)}

        But got tensor:

        #{inspect(data)}
        """
      end

      data_and_shapes =
        if client.platform == :host do
          %EXLA.Shape{dtype: {:tuple, shapes}} = send_shape
          Enum.zip(nx_to_io(data), shapes)
        else
          [{nx_to_io(data), send_shape}]
        end

      pred = EXLA.Shape.make_shape({:pred, 8}, {})
      :ok = EXLA.Client.to_infeed(client, device_id, [{<<1::8-native>>, pred} | data_and_shapes])
    end

    defp nx_to_io(%Nx.Tensor{} = tensor),
      do: [Nx.to_binary(tensor)]

    defp nx_to_io(map) when is_map(map),
      do: map |> Enum.sort() |> Enum.flat_map(fn {_, v} -> nx_to_io(v) end)

    defp nx_to_io(tuple) when is_tuple(tuple),
      do: tuple |> Tuple.to_list() |> Enum.flat_map(&nx_to_io/1)

    defp nx_to_io(other),
      do: [other |> Nx.to_tensor() |> Nx.to_binary()]

    def recv(%{pid: pid, outfeed: outfeed, lock: lock, recv: recv, recv_shapes: shapes}) do
      if pid != self() do
        raise "EXLA streams require recv to be called from the process that started the stream"
      end

      unless Process.alive?(outfeed) do
        raise "cannot recv from stream because it has been terminated"
      end

      buffers =
        for shape <- shapes do
          receive do
            {^lock, binary} -> %EXLA.Buffer{data: binary, shape: shape}
          end
        end

      EXLA.Defn.Buffer.to_nx!(buffers, recv)
    end

    def done(%{lock: lock, pid: pid, ref: ref, keep_on_device: keep_on_device, done: done}) do
      if pid != self() do
        raise "EXLA streams require recv to be called from the process that started the stream"
      end

      # This will send message to stop the loop
      # and set the lock to the output process.
      EXLA.Defn.Lock.unlock(lock)

      receive do
        {^lock, _} ->
          raise "cannot mark stream as done when there are recv messages pending"

        {^ref, result} ->
          tensors = EXLA.Defn.Buffer.to_nx!(result, done)
          if keep_on_device, do: tensors, else: Nx.backend_transfer(tensors)
      end
    end
  end
end
