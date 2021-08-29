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

      %{client: client, device_id: device_id} = executable
      pred = EXLA.Shape.make_shape({:pred, 8}, {})
      :ok = EXLA.Client.to_infeed(client, device_id, <<1::8-native>>, pred)

      if client.platform == :host do
        %EXLA.Shape{dtype: {:tuple, shapes}} = send_shape

        Enum.zip_with(shapes, nx_to_io(data), fn shape, binary ->
          :ok = EXLA.Client.to_infeed(client, device_id, binary, shape)
        end)
      else
        :ok = EXLA.Client.to_infeed(client, device_id, nx_to_io(data), send_shape)
      end

      :ok
    end

    defp nx_to_io(%Nx.Tensor{} = tensor),
      do: [tensor |> Nx.backend_transfer() |> Nx.to_binary()]

    defp nx_to_io(map) when is_map(map),
      do: map |> Enum.sort() |> Enum.flat_map(fn {_, v} -> nx_to_io(v) end)

    defp nx_to_io(tuple) when is_tuple(tuple),
      do: tuple |> Tuple.to_list() |> Enum.flat_map(&nx_to_io/1)

    defp nx_to_io(other),
      do: [other |> Nx.to_tensor() |> Nx.to_binary()]

    def recv(%{executable: executable, recv_shape: recv_shape}) do
      %EXLA.Shape{dtype: {:tuple, shapes}} = recv_shape
      %{client: client, device_id: device_id} = executable

      for shape <- shapes do
        EXLA.Client.from_outfeed(client, device_id, shape)
      end
    end

    def done(%{executable: executable, keep_on_device: keep_on_device, done: done}) do
      pred = EXLA.Shape.make_shape({:pred, 8}, {})
      :ok = EXLA.Client.to_infeed(executable.client, executable.device_id, <<0::8-native>>, pred)
      if keep_on_device, do: done.(), else: Nx.backend_transfer(done.())
    end
  end
end
