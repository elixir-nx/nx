defmodule EXLA.Client do
  @moduledoc """
  Functions for managing `EXLA.Client`.

  See `EXLA` module docs for a general introduction.
  """
  require Logger

  @doc false
  use GenServer

  @name __MODULE__

  @enforce_keys [:ref, :platform, :name, :device_count, :default_device_id, :automatic_transfers]
  defstruct [:ref, :platform, :name, :device_count, :default_device_id, :automatic_transfers]

  @doc """
  Returns the name of the default client.
  """
  def default_name do
    case Application.fetch_env(:exla, :default_client) do
      {:ok, client} ->
        client

      :error ->
        all_clients = Application.fetch_env!(:exla, :clients)
        preferred_clients = Application.fetch_env!(:exla, :preferred_clients)
        supported_platforms = get_supported_platforms()

        client =
          Enum.find(preferred_clients, fn client ->
            config =
              all_clients[client] ||
                raise "unknown client #{inspect(client)} given as :preferred_clients. " <>
                        "If you plan to use :cuda or :rocm, make sure the XLA_TARGET environment variable " <>
                        "is appropriately set. Currently it is set to #{inspect(System.get_env("XLA_TARGET"))}"

            platform = config[:platform] || :host

            # If you install XLA with CUDA/ROCm but there are no devices,
            # we don't want to pick them by default.
            match?(%{^platform => devices} when devices >= 0, supported_platforms)
          end)

        unless client do
          raise(
            "could not find default client for EXLA. The configured EXLA clients are: #{inspect(all_clients)}"
          )
        end

        Application.put_env(:exla, :default_client, client)
        client
    end
  end

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
    EXLA.NIF.get_supported_platforms()
  end

  @doc """
  Sends `data_and_typespecs` to device infeed.

  `data_and_typespecs` is a list of values corresponding to a single
  infeed operation. It must be a list of two element tuples where the
  first element is a binary and the second element is a `EXLA.Typespec`.
  """
  def to_infeed(%EXLA.Client{ref: client}, device_id, data_and_typespecs)
      when is_list(data_and_typespecs) do
    {buffers, typespecs} = Enum.unzip(data_and_typespecs)
    EXLA.NIF.transfer_to_infeed(client, device_id, buffers, typespecs)
  end

  @doc """
  Sends buffer from device outfeed to the given process tagged by `ref`.
  """
  def from_outfeed(%EXLA.Client{ref: client}, device_id, typespecs, pid, ref)
      when is_list(typespecs) do
    EXLA.NIF.transfer_from_outfeed(client, device_id, typespecs, pid, ref)
  end

  @doc """
  Returns the memory statistics for a client.

  Returns a map with the following keys:
    * `:allocated` - Currently allocated memory in bytes
    * `:peak` - Peak memory usage in bytes
    * `:per_device` - Map of device ID to allocated memory in bytes

  ## Examples

      client = EXLA.Client.fetch!(:host)
      EXLA.Client.get_memory_statistics(client)
      %{allocated: 0, peak: 0, per_device: %{0 => 0}}

  """
  def get_memory_statistics(%__MODULE__{ref: ref}) do
    %{
      allocated: EXLA.NIF.get_allocated_memory(ref),
      peak: EXLA.NIF.get_peak_memory(ref),
      per_device: EXLA.NIF.get_per_device_memory(ref)
    }
  end

  @doc """
  Resets the peak memory counter for a client.

  ## Examples

      iex> client = EXLA.Client.fetch!(:host)
      iex> EXLA.Client.reset_peak_memory(client)
      :ok

  """
  def reset_peak_memory(%__MODULE__{ref: ref}) do
    EXLA.NIF.reset_peak_memory(ref)
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
    platform = Keyword.get(options, :platform)
    memory_fraction = Keyword.get(options, :memory_fraction, 0.9)
    preallocate = Keyword.get(options, :preallocate, true)
    platforms = Map.keys(EXLA.Client.get_supported_platforms())

    ref =
      case platform do
        nil ->
          Logger.debug("""
          No platform configuration specified, falling back to host platform
          Available platforms are: #{inspect(platforms)}
          """)

          EXLA.NIF.get_host_client()

        :host ->
          EXLA.NIF.get_host_client()

        :cuda ->
          EXLA.NIF.get_gpu_client(memory_fraction, preallocate)

        :rocm ->
          EXLA.NIF.get_gpu_client(memory_fraction, preallocate)

        :tpu ->
          EXLA.NIF.get_tpu_client()

        _ ->
          raise ArgumentError, "unknown EXLA platform: #{inspect(platform)}"
      end

    device_count = EXLA.NIF.get_device_count(ref)
    default_device_id = Keyword.get(options, :default_device_id, 0)

    if default_device_id not in 0..(device_count - 1) do
      raise ArgumentError, ":default_device_id must be a number between 0 and #{device_count - 1}"
    end

    automatic_transfers = Keyword.get(options, :automatic_transfers, platform == :host)

    %EXLA.Client{
      ref: ref,
      platform: platform,
      name: name,
      device_count: device_count,
      default_device_id: default_device_id,
      automatic_transfers: automatic_transfers
    }
  end
end
