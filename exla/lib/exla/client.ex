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
      # This is the default num_replicas used on compile, so we should probably rename it
      number_of_replicas = Keyword.get(options, :number_of_replicas, 1)
      # The number of threads that will get used during a computation
      intra_op_parallelism_threads = Keyword.get(options, :intra_op_parallelism_threads, -1)

      ref =
        case platform do
          :host -> EXLA.NIF.get_host_client(number_of_replicas, intra_op_parallelism_threads)
          :cuda -> EXLA.NIF.get_cuda_client(number_of_replicas, intra_op_parallelism_threads)
          :rocm -> EXLA.NIF.get_rocm_client(number_of_replicas, intra_op_parallelism_threads)
          _ -> raise ArgumentError, "unknown Exla platform: #{inspect(platform)}"
        end
        |> unwrap!()

      device_count = EXLA.NIF.get_device_count(ref) |> unwrap!()

      default_device_ordinal =
        if default = options[:default_device_ordinal] do
          default
        else
          EXLA.NIF.get_default_device_ordinal(ref) |> unwrap!()
        end

      client = %Client{
        ref: ref,
        platform: platform,
        name: name,
        device_count: device_count,
        default_device_ordinal: default_device_ordinal
      }

      {nil, client}
    end)
  end

  @doc false
  def check_device_compatibility!(
        %Client{device_count: device_count, default_device_ordinal: default_device_ordinal},
        ordinal
      ) when is_integer(ordinal) do
    cond do
      ordinal < 0 ->
        default_device_ordinal

      ordinal < device_count ->
        ordinal

      true ->
        raise ArgumentError, "Invalid device ordinal."
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
