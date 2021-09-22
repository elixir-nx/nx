defmodule EXLA.Client do
  @moduledoc """
  Functions for managing `EXLA.Client`.

  See `EXLA` module docs for a general introduction.
  """

  use GenServer
  @name __MODULE__

  @enforce_keys [:ref, :platform, :name, :device_count, :devices, :default_device_id]
  defstruct [:ref, :platform, :name, :device_count, :devices, :default_device_id]

  @doc """
  Fetches a client with the given `name` from configuration.
  """
  def fetch!(name) when is_atom(name) do
    # We could use the LockedCache but that is ETS based and the clients
    # are static enough that we can keep them on `persistent_term`.
    :persistent_term.get({__MODULE__, name}, nil) ||
      (
        clients = Application.fetch_env!(:exla, :clients)

        options =
          Keyword.get(clients, name) ||
            raise ArgumentError,
                  "could not find EXLA client named #{inspect(name)}, the clients specified " <>
                    "in your config files are: #{inspect(Keyword.keys(clients))}"

        GenServer.call(@name, {:client, name, options}, :infinity)
      )
  end

  @doc """
  Returns a map of supported platforms with device information.
  """
  def get_supported_platforms do
    EXLA.NIF.get_supported_platforms() |> unwrap!()
  end

  @doc """
  Sends data to device infeed.

  Data must be a VM binary or a flat list of VM binaries.

  > Note: XLA does not support tuple infeed shapes when running on
  > host. Passing one will simply block the operation indefinitely.
  > Instead, convert the tuple into multiple infeed operations.
  """
  def to_infeed(%EXLA.Client{ref: client}, device_id, data, %EXLA.Shape{ref: shape})
      when is_binary(data) do
    EXLA.NIF.transfer_to_infeed(client, device_id, [data], shape) |> unwrap!()
  end

  def to_infeed(%EXLA.Client{}, _device_id, [], %EXLA.Shape{}) do
    :ok
  end

  def to_infeed(%EXLA.Client{ref: client}, device_id, [data | _] = list, %EXLA.Shape{ref: shape})
      when is_binary(data) do
    EXLA.NIF.transfer_to_infeed(client, device_id, list, shape) |> unwrap!()
  end

  @doc """
  Retrieves buffer from device outfeed.

  If you want to receive a tuple shape, consider using `from_tuple_outfeed/2`,
  as it will return a separate binary for each individual tuple element.

  > Note: XLA does not support tuple outfeed shapes. Passing one will simply
  > block the operation indefinitely. Instead, convert the tuple into multiple
  > outfeed operations.
  """
  def from_outfeed(%EXLA.Client{ref: client}, device_id, %EXLA.Shape{ref: shape_ref}) do
    EXLA.NIF.transfer_from_outfeed(client, device_id, shape_ref) |> unwrap!()
  end

  ## Callbacks

  @doc false
  def start_link(_) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @impl true
  def init(:ok) do
    {:ok, :unused_state}
  end

  @impl true
  def handle_call({:client, name, options}, _from, state) do
    client = :persistent_term.get({__MODULE__, name}, nil) || build_client(name, options)
    :persistent_term.put({__MODULE__, name}, client)
    {:reply, client, state}
  end

  defp build_client(name, options) do
    platform = Keyword.get(options, :platform, :host)
    default_device_id = Keyword.get(options, :default_device_id, 0)
    memory_fraction = Keyword.get(options, :memory_fraction, 0.9)

    preallocate = Keyword.get(options, :preallocate, true)
    preallocate_int = if preallocate, do: 1, else: 0

    ref =
      case platform do
        :host -> EXLA.NIF.get_host_client()
        :cuda -> EXLA.NIF.get_gpu_client(memory_fraction, preallocate_int)
        :rocm -> EXLA.NIF.get_gpu_client(memory_fraction, preallocate_int)
        :tpu -> EXLA.NIF.get_tpu_client()
        _ -> raise ArgumentError, "unknown EXLA platform: #{inspect(platform)}"
      end
      |> unwrap!()

    device_count = EXLA.NIF.get_device_count(ref) |> unwrap!()
    devices = EXLA.NIF.get_devices(ref) |> unwrap!()

    if default_device_id not in 0..(device_count - 1) do
      raise ArgumentError, ":default_device_id must be a number between 0 and #{device_count - 1}"
    end

    %EXLA.Client{
      ref: ref,
      platform: platform,
      name: name,
      device_count: device_count,
      devices: devices,
      default_device_id: default_device_id
    }
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
