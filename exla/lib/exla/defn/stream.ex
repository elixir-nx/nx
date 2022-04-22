defmodule EXLA.Defn.Stream do
  @moduledoc false

  keys =
    [:lock, :outfeed, :pid, :runner, :send, :recv, :send_shape] ++
      [:recv_length, :done, :client, :device_id]

  @derive {Inspect, only: [:pid, :client, :device_id, :send, :recv]}
  @enforce_keys keys
  defstruct keys

  def run(
        executable,
        lock,
        runner,
        outfeed,
        send,
        send_shape,
        recv,
        recv_shapes,
        done
      ) do
    %{client: client, device_id: device_id} = executable

    # With the task and outfeed in place, we now register the unlock callback:
    # if the current process shuts down, we send an infeed to stop the loop,
    # and then we block until the outfeed completes.
    ^lock =
      EXLA.Defn.Lock.on_unlock(
        lock,
        fn -> send(runner, lock) end,
        fn -> halt_stream(client, device_id, outfeed) end
      )

    %EXLA.Defn.Stream{
      pid: self(),
      runner: runner,
      outfeed: outfeed,
      lock: lock,
      send: send,
      send_shape: send_shape,
      recv: recv,
      recv_length: length(recv_shapes),
      client: client,
      device_id: device_id,
      done: done
    }
  end

  # It is time to halt the stream, we do it by sending 0 for the loop infeed.
  # Then we wait for the outfeed process to read all.
  defp halt_stream(client, device_id, outfeed) do
    pred = EXLA.Shape.make_shape({:pred, 8}, {})
    :ok = EXLA.Client.to_infeed(client, device_id, [{<<0::8-native>>, pred}])
    {:transfer, outfeed}
  end

  defimpl Nx.Stream do
    def send(
          %{pid: pid, client: client, device_id: device_id, send: send, send_shape: send_shape},
          data
        ) do
      if pid != self() do
        raise "EXLA streams require recv to be called from the process that started the stream"
      end

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

    defp nx_to_io(container) do
      container
      |> Nx.Defn.Composite.reduce([], fn tensor, acc ->
        [tensor |> Nx.to_tensor() |> Nx.to_binary() | acc]
      end)
      |> Enum.reverse()
    end

    def recv(%{pid: pid, outfeed: outfeed, lock: lock, recv: recv, recv_length: length}) do
      if pid != self() do
        raise "EXLA streams require recv to be called from the process that started the stream"
      end

      unless Process.alive?(outfeed) do
        raise "cannot recv from stream because it has been terminated"
      end

      buffers =
        for _ <- 1..length//1 do
          receive do
            {^lock, binary} -> binary
          end
        end

      EXLA.Defn.Buffers.to_nx!(buffers, recv)
    end

    def done(%{
          lock: lock,
          outfeed: outfeed,
          pid: pid,
          runner: runner,
          done: done
        }) do
      if pid != self() do
        raise "EXLA streams require recv to be called from the process that started the stream"
      end

      # This will write to infeed to stop the loop. We know unlocking
      # is race free because we can only write to infeed from this process
      # (or it is automatically written if this process is dead).
      #
      # Once we unlock, the lock process will now wait until the outfeed
      # terminates.
      EXLA.Defn.Lock.unlock(lock)

      # We also wait until the outfeed completes to ensure
      # all output has been consumed before we return.
      outfeed_ref = Process.monitor(outfeed)

      receive do
        {^lock, _} ->
          raise "cannot mark stream as done when there are recv messages pending"

        {:DOWN, ^outfeed_ref, _, _, _} ->
          [result] = EXLA.Defn.Runner.read(runner)
          EXLA.Defn.Buffers.to_nx!(result, done)
      end
    end
  end
end
