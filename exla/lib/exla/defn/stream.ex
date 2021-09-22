defmodule EXLA.Defn.Stream do
  @moduledoc false

  # If it terminates, another one will be started dynamically
  use GenServer, restart: :transient

  @derive {Inspect, only: [:client, :device_id, :keep_on_device, :send, :recv]}
  @enforce_keys [
    :send,
    :send_shape,
    :recv,
    :recv_shape,
    :done,
    :client,
    :device_id,
    :keep_on_device
  ]
  defstruct [:send, :send_shape, :recv, :recv_shape, :done, :client, :device_id, :keep_on_device]

  @registry EXLA.Registry
  @supervisor EXLA.DynamicSupervisor

  @doc false
  def handler(client, device_id) do
    case Registry.lookup(@registry, {client.name, device_id}) do
      [{pid, _}] ->
        pid

      [] ->
        case DynamicSupervisor.start_child(@supervisor, {__MODULE__, {client, device_id}}) do
          {:ok, pid} -> pid
          {:error, {:already_started, pid}} -> pid
        end
    end
  end

  @doc false
  def start_link({client, device_id}) do
    name = {:via, Registry, {@registry, {client.name, device_id}}}
    GenServer.start_link(__MODULE__, {client, device_id}, name: name)
  end

  @impl true
  def init({client, device_id}) do
    {:ok, %{client: client, device_id: device_id}}
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

    def recv(%{client: client, device_id: device_id, recv_shape: recv_shape}) do
      %EXLA.Shape{dtype: {:tuple, shapes}} = recv_shape

      for shape <- shapes do
        EXLA.Client.from_outfeed(client, device_id, shape)
      end
    end

    def done(%{client: client, device_id: device_id, keep_on_device: keep_on_device, done: done}) do
      pred = EXLA.Shape.make_shape({:pred, 8}, {})
      :ok = EXLA.Client.to_infeed(client, device_id, <<0::8-native>>, pred)
      if keep_on_device, do: done.(), else: Nx.backend_transfer(done.())
    end
  end
end
