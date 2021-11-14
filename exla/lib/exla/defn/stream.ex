defmodule EXLA.Defn.Stream do
  @moduledoc false

  keys =
    [:lock, :outfeed, :pid, :ref, :send, :send_shape] ++
      [:recv_shapes, :done, :client, :device_id, :keep_on_device]

  @derive {Inspect, only: [:pid, :client, :device_id, :keep_on_device, :send]}
  @enforce_keys keys
  defstruct keys

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

    def recv(%{pid: pid, outfeed: outfeed, lock: lock, recv_shapes: shapes}) do
      if pid != self() do
        raise "EXLA streams require recv to be called from the process that started the stream"
      end

      unless Process.alive?(outfeed) do
        raise "cannot recv from stream because it has been terminated"
      end

      for _ <- shapes do
        receive do
          {^lock, binary} -> binary
        end
      end
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
