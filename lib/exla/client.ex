defmodule Exla.Client do
  @moduledoc """
  Functions for managing `Exla.Client`.

  See `Exla` module docs for a general introduction.
  """

  alias __MODULE__
  alias Exla.{Computation, Executable, Shape}

  @enforce_keys [:ref, :platform, :name, :device_count, :default_device_ordinal]
  defstruct [:ref, :platform, :name, :device_count, :default_device_ordinal]

  def fetch!(name) do
    Exla.LockedCache.run({__MODULE__, name}, fn ->
      clients = Application.fetch_env!(:exla, :clients)

      options =
        Keyword.get(clients, name) ||
          raise ArgumentError,
                "could not find Exla client named #{inspect(name)}, the clients specified " <>
                  "in your config files are: #{inspect(Keyword.keys(clients))}"

      platform = Keyword.get(options, :platform, :host)
      number_of_replicas = Keyword.get(options, :number_of_replicas, 1)
      intra_op_parallelism_threads = Keyword.get(options, :intra_op_parallelism_threads, -1)

      ref =
        case platform do
          :host -> Exla.NIF.get_host_client(number_of_replicas, intra_op_parallelism_threads)
          :cuda -> Exla.NIF.get_cuda_client(number_of_replicas, intra_op_parallelism_threads)
          :rocm -> Exla.NIF.get_rocm_client(number_of_replicas, intra_op_parallelism_threads)
          _ -> raise ArgumentError, "unknown Exla platform: #{inspect(platform)}"
        end
        |> unwrap!()

      device_count = Exla.NIF.get_device_count(ref) |> unwrap!()
      default_device_ordinal = Exla.NIF.get_default_device_ordinal(ref) |> unwrap!()

      %Client{
        ref: ref,
        platform: platform,
        name: name,
        device_count: device_count,
        default_device_ordinal: default_device_ordinal
      }
    end)
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))

  def compile(
        client = %Client{},
        computation = %Computation{output_shape: output_shape},
        argument_shapes,
        options \\ []
      ) do
    device_ordinal = Keyword.get(options, :device_ordinal, -1)
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)
    # This needs to be investigated a bit more
    use_spmd = Keyword.get(options, :use_spmd, false)
    use_spmd_int = if use_spmd, do: 1, else: 0
    device_ordinal = check_device_compatibility!(client, device_ordinal)

    shape_refs =
      argument_shapes
      |> Enum.map(& Shape.shard(&1, num_replicas*num_partitions))
      |> Enum.map(& &1.ref)

    # Executable Build Context
    # TODO: Validate replicas, partitions, and shapes
    IO.inspect device_ordinal

    ref =
      Exla.NIF.compile(
        client.ref,
        computation.ref,
        shape_refs,
        device_ordinal,
        num_replicas,
        num_partitions,
        use_spmd_int
      )
      |> unwrap!

    %Executable{
      client: client,
      ref: ref,
      output_shape: output_shape,
      device_ordinal: device_ordinal,
      num_replicas: num_replicas,
      num_partitions: num_partitions
    }
  end

  def check_device_compatibility!(
        %Client{device_count: device_count, default_device_ordinal: default_device_ordinal},
        ordinal
      ) do
    cond do
      ordinal < 0 ->
        -1

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
    {:ok, platforms} = Exla.NIF.get_supported_platforms()
    platforms
  end
end
