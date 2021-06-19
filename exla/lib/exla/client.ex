defmodule EXLA.Client do
  @moduledoc """
  Functions for managing `Exla.Client`.

  See `Exla` module docs for a general introduction.
  """

  alias __MODULE__

  @enforce_keys [:ref, :platform, :name, :device_count, :devices]
  defstruct [:ref, :platform, :name, :device_count, :devices]

  @doc """
  Fetches a client with the given `name` from configuration.
  """
  def fetch!(name) do
    {_, client} = fetch_client!(name)
    client
  end

  defp fetch_client!(name) do
    EXLA.LockedCache.run({__MODULE__, name}, fn ->
      clients = Application.fetch_env!(:exla, :clients)

      options =
        Keyword.get(clients, name) ||
          raise ArgumentError,
                "could not find EXLA client named #{inspect(name)}, the clients specified " <>
                  "in your config files are: #{inspect(Keyword.keys(clients))}"

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

      client = %Client{
        ref: ref,
        platform: platform,
        name: name,
        device_count: device_count,
        devices: devices
      }

      {nil, client}
    end)
  end

  @doc """
  Returns a map of supported platforms with device information.
  """
  def get_supported_platforms do
    EXLA.NIF.get_supported_platforms() |> unwrap!()
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
