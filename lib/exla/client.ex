defmodule EXLA.Client do
  @moduledoc """
  Functions for managing `Exla.Client`.

  See `Exla` module docs for a general introduction.
  """

  alias __MODULE__
  alias EXLA.{Computation, Executable, Shape}

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
      number_of_replicas = Keyword.get(options, :number_of_replicas, 1)
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
    # TODO: investigated a bit more
    use_spmd = Keyword.get(options, :use_spmd, false)
    use_spmd_int = if use_spmd, do: 1, else: 0
    device_ordinal = check_device_compatibility!(client, device_ordinal)

    shape_refs =
      argument_shapes
      |> Enum.map(&Shape.shard(&1, num_replicas * num_partitions))
      |> Enum.map(& &1.ref)

    # Executable Build Context
    # TODO: Validate replicas, partitions, and shapes

    ref =
      EXLA.NIF.compile(
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
  Awaits for all running streams on the given device.
  """
  def await_streams(%Client{ref: ref, platform: platform} = client, buffer, keep_on_device) do
    # See https://github.com/elixir-nx/exla/pull/124, for discussion on this
    case platform do
      :host ->
        EXLA.NIF.await_streams_cpu(ref, buffer, keep_on_device)
      _ ->
        EXLA.NIF.await_streams_io(ref, buffer, keep_on_device)
    end
  end

  @doc """
  Returns a map of supported platforms with device information.
  """
  def get_supported_platforms do
    {:ok, platforms} = EXLA.NIF.get_supported_platforms()
    platforms
  end
end
