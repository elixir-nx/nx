defmodule EXLA.Defn.Stream do
  @moduledoc false

  # If it terminates, another one will be started dynamically
  use GenServer, restart: :transient

  @derive {Inspect, only: [:client, :device_id, :keep_on_device, :send, :recv]}
  @enforce_keys [
    :lock,
    :send,
    :send_shape,
    :recv,
    :recv_shape,
    :done,
    :client,
    :device_id,
    :keep_on_device
  ]
  defstruct [
    :lock,
    :send,
    :send_shape,
    :recv,
    :recv_shape,
    :done,
    :client,
    :device_id,
    :keep_on_device
  ]

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

    def recv(%{client: client, device_id: device_id, recv_shape: recv_shape}) do
      # TODO: Move this to a separate process
      true = get_flag(client, device_id) == 1

      %EXLA.Shape{dtype: {:tuple, shapes}} = recv_shape
      ref = make_ref()
      :ok = EXLA.Client.from_outfeed(client, device_id, shapes, self(), ref)

      for _ <- shapes do
        receive do
          {^ref, binary} -> binary
        end
      end
    end

    def done(%{
          lock: lock,
          client: client,
          device_id: device_id,
          keep_on_device: keep_on_device,
          done: done
        }) do
      EXLA.Lock.unlock(lock)
      # TODO: Move this to a separate process
      true = get_flag(client, device_id) == 0
      if keep_on_device, do: done.(), else: Nx.backend_transfer(done.())
    end

    defp get_flag(client, device_id) do
      ref = make_ref()
      flag_shape = EXLA.Shape.make_shape({:u, 16}, {})
      :ok = EXLA.Client.from_outfeed(client, device_id, [flag_shape], self(), ref)

      receive do
        {^ref, <<flag::native-unsigned-16>>} -> flag
      end
    end
  end
end
