defmodule Exla.Client do
  @moduledoc """
  Functions for managing Exla.Client.

  See `Exla` module docs for a general introduction.
  """

  alias __MODULE__, as: Client
  alias Exla.Computation
  alias Exla.Executable
  @enforce_keys [:ref, :platform]
  defstruct [:ref, :platform]

  def fetch!(name) do
    Exla.LockedCache.run({__MODULE__, name}, fn ->
      clients = Application.fetch_env!(:exla, :clients)

      config =
        Keyword.get(clients, name) ||
          raise ArgumentError, "could not find Exla client named #{inspect(name)}, " <>
            "the clients specified in your config files are: #{inspect(Keyword.keys(clients))}"

      create_client(config)
    end)
  end

  # TODO: To go along with some of the discussion in: https://github.com/seanmor5/exla/pull/12
  # The Python XLA API offers 3 additional methods for client creation:
  # `get_cpu_client`, `get_nvidia_gpu_client`, and `get_tpu_client`. They essentially
  # wrap the method below with preset configurations, allocators, etc. that work out
  # of the box with CPU/GPU/TPU respectively. This has the benefit of giving the user
  # a guaranteed working client without having to mess around with specifying a device,
  # allocator, etc. For example, the current Naive Allocator as it's set up and configured
  # doesn't work with GPU. We would need to set up some special configurations for that
  # to work. We can give the user the ability to fully customize their setup around this
  # function, but also offer the more convenient and safer `get_[device]_client` methods.
  # Alternatively, we can keep this method private, and only expose the 3 client device
  # creation methods, with limited, but safer configuration options.
  # TODO: Make this function private
  def create_client(options \\ []) do
    # TODO: Rename this function to get_local_client. It is a singleton,
    # non-thread-safe resource in XLA so we need to mimic the same
    # in Elixir. We should also have distinct steps for configuring and for
    # getting it. See: https://github.com/seanmor5/exla/pull/12
    platform = Keyword.get(options, :platform, :host)
    number_of_replicas = Keyword.get(options, :number_of_replicas, 1)
    intra_op_parallelism_threads = Keyword.get(options, :intra_op_parallelism_threads, -1)

    # TODO: Add MapSet of allowed devices to block device use in a multi-device context
    {:ok, ref} =
      case platform do
        :host -> Exla.NIF.get_cpu_client(number_of_replicas, intra_op_parallelism_threads)
        :cuda -> Exla.NIF.get_gpu_client(number_of_replicas, intra_op_parallelism_threads)
      end

    %Client{ref: ref, platform: platform}
  end

  # TODO: These methods are only called once, so for efficiency we can run them when the client is created
  def get_default_device_ordinal(%Client{ref: client}) do
    {:ok, ordinal} = Exla.NIF.get_default_device_ordinal(client)
    ordinal
  end

  def get_device_count(%Client{ref: client}) do
    {:ok, count} = Exla.NIF.get_device_count(client)
    count
  end

  def compile(
         client = %Client{ref: ref},
         computation = %Computation{output_shape: output_shape},
         argument_shapes,
         options \\ []
       ) do
    device_ordinal = Keyword.get(options, :device_ordinal, -1)
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)

    shape_refs =
      argument_shapes
      |> Enum.map(& &1.ref)

    # Executable Build Context
    # TODO: Validate replicas, partitions, and shapes
    # TODO: Use device
    with {:ok, {_platform, device_ordinal}} <-
           check_device_compatibility(client, {client.platform, device_ordinal}),
         {:ok, ref} <-
           Exla.NIF.compile(
             ref,
             computation.ref,
             shape_refs,
             device_ordinal,
             num_replicas,
             num_partitions
           ) do
      %Executable{
        client: client,
        ref: ref,
        output_shape: output_shape,
        device: {client.platform, device_ordinal}
      }
    end
  end

  defp check_device_compatibility(
        client = %Client{platform: platform},
        {platform, ordinal}
      ) do
    cond do
      ordinal < 0 ->
        {:ok, {platform, Client.get_default_device_ordinal(client)}}

      ordinal < Client.get_device_count(client) ->
        {:ok, {platform, ordinal}}

      true ->
        {:error, "Invalid device ordinal."}
    end
  end
end
