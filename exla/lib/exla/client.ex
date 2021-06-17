defmodule EXLA.Client do
  @moduledoc """
  Functions for managing `Exla.Client`.

  See `Exla` module docs for a general introduction.
  """

  alias __MODULE__

  @enforce_keys [:ref, :platform, :name, :device_count, :default_device_ordinal]
  defstruct [:ref, :platform, :name, :device_count, :default_device_ordinal]

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
          :cuda -> EXLA.NIF.get_cuda_client(memory_fraction, preallocate_int)
          :tpu -> EXLA.NIF.get_tpu_client()
          #         :rocm -> EXLA.NIF.get_rocm_client(num_replicas, intra_op_parallelism_threads, memory_fraction, preallocate_int)
          _ -> raise ArgumentError, "unknown Exla platform: #{inspect(platform)}"
        end
        |> unwrap!()

      #      device_count = EXLA.NIF.get_device_count(ref) |> unwrap!()

      client = %Client{
        ref: ref,
        platform: platform,
        name: name,
        device_count: 1,
        default_device_ordinal: 0
      }

      {nil, client}
    end)
  end

  @doc """
  Validates the given device ordinal.
  """
  def validate_device_ordinal!(
        %Client{device_count: device_count, default_device_ordinal: default_device_ordinal},
        ordinal
      )
      when is_integer(ordinal) do
    cond do
      ordinal < 0 ->
        default_device_ordinal

      ordinal < device_count ->
        ordinal

      true ->
        raise ArgumentError,
              "Invalid device ordinal. It must be -1, to pick the default, " <>
                "or a number >= 0 and < #{device_count}"
    end
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
