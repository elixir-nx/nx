defmodule EXLA.Client do
  @moduledoc """
  Functions for managing `EXLA.Client`.

  See `EXLA` module docs for a general introduction.
  """

  use GenServer
  @name __MODULE__

  @enforce_keys [:ref, :platform, :name, :device_count, :devices]
  defstruct [:ref, :platform, :name, :device_count, :devices]

  @doc """
  Fetches a client with the given `name` from configuration.
  """
  def fetch!(name) do
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

  ## Callbacks

  @doc false
  def start_link(_) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @doc false
  def init(:ok) do
    {:ok, :unused_state}
  end

  @doc false
  def handle_call({:client, name, options}, _from, state) do
    client = :persistent_term.get({__MODULE__, name}, nil) || build_client(name, options)
    :persistent_term.put({__MODULE__, name}, client)
    {:reply, client, state}
  end

  defp build_client(name, options) do
    platform = Keyword.get(options, :platform, :host)

    # Fraction of GPU memory to preallocate
    memory_fraction = Keyword.get(options, :memory_fraction, 0.9)

    # Flag for preallocating GPU memory
    preallocate = Keyword.get(options, :preallocate, true)
    preallocate_int = if preallocate, do: 1, else: 0

    ref =
      case platform do
        :host -> EXLA.NIF.get_host_client()
        :cuda -> EXLA.NIF.get_gpu_client(memory_fraction, preallocate_int)
        :rocm -> EXLA.NIF.get_gpu_client(memory_fraction, preallocate_int)
        :tpu -> EXLA.NIF.get_tpu_client()
        _ -> raise ArgumentError, "unknown Exla platform: #{inspect(platform)}"
      end
      |> unwrap!()

    device_count = EXLA.NIF.get_device_count(ref) |> unwrap!()
    devices = EXLA.NIF.get_devices(ref) |> unwrap!()

    %EXLA.Client{
      ref: ref,
      platform: platform,
      name: name,
      device_count: device_count,
      devices: devices
    }
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
