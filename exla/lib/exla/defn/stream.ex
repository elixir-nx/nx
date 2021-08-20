defmodule EXLA.Defn.Stream do
  @moduledoc false
  @derive {Inspect, only: [:executable, :keep_on_device, :send, :recv]}
  @enforce_keys [:send, :send_shape, :recv, :recv_shape, :done, :executable, :keep_on_device]
  defstruct [:send, :send_shape, :recv, :recv_shape, :done, :executable, :keep_on_device]

  defimpl Nx.Stream do
    def send(%{executable: executable, send: send, send_shape: send_shape}, data) do
      unless Nx.compatible?(send, data) do
        raise ArgumentError, """
        Nx stream expected a tensor of type, shape, and names on send:

        #{inspect(send)}

        But got tensor:

        #{inspect(data)}
        """
      end

      pred = EXLA.Shape.make_shape({:pred, 8}, {})
      :ok = EXLA.Client.to_infeed(executable.client, executable.device_id, <<0::8-native>>, pred)

      # TODO: Allow a list of binaries to be given to infeed
      binary = data |> nx_to_binary() |> IO.iodata_to_binary()
      :ok = EXLA.Client.to_infeed(executable.client, executable.device_id, binary, send_shape)
    end

    defp nx_to_binary(%Nx.Tensor{} = tensor) do
      [tensor |> Nx.backend_transfer() |> Nx.to_binary()]
    end

    defp nx_to_binary(map) when is_map(map) do
      map |> Enum.sort() |> Enum.flat_map(fn {_, v} -> nx_to_binary(v) end)
    end

    defp nx_to_binary(tuple) when is_tuple(tuple) do
      Enum.flat_map(tuple, &nx_to_binary/1)
    end

    defp nx_to_binary(other) do
      [other |> Nx.to_tensor() |> Nx.to_binary()]
    end

    def recv(%{executable: executable, recv_shape: recv_shape}) do
      # TODO: Decode binary into tensors
      EXLA.Client.from_outfeed(executable.client, executable.device_id, recv_shape)
    end

    def done(%{executable: executable, keep_on_device: keep_on_device, done: done}) do
      pred = EXLA.Shape.make_shape({:pred, 8}, {})
      :ok = EXLA.Client.to_infeed(executable.client, executable.device_id, <<0::8-native>>, pred)
      if keep_on_device, do: done.(), else: Nx.backend_transfer(done.())
    end
  end
end
