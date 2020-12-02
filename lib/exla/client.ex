defmodule Exla.Client do
  @moduledoc """
  Functions for managing `Exla.Client`.

  See `Exla` module docs for a general introduction.
  """

  alias __MODULE__
  alias Exla.{Computation, Executable}

  @enforce_keys [:ref, :platform, :name]
  defstruct [:ref, :platform, :name]

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
        end
        |> unwrap!()

      %Client{ref: ref, platform: platform, name: name}
    end)
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))

  # TODO: The Python XLA API offers 3 additional methods for client creation:
  # `get_host_client`, `get_nvidia_gpu_client`, and `get_tpu_client`. They essentially
  # wrap the method below with preset configurations, allocators, etc. that work out
  # of the box with CPU/GPU/TPU respectively. This has the benefit of giving the user
  # a guaranteed working client without having to mess around with specifying a device,
  # allocator, etc. For example, the current Naive Allocator as it's set up and configured
  # doesn't work with GPU. We would need to set up some special configurations for that
  # to work. We can give the user the ability to fully customize their setup around this
  # function, but also offer the more convenient and safer `get_[device]_client` methods.
  # Alternatively, we can keep this method private, and only expose the 3 client device
  # creation methods, with limited, but safer configuration options.

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
        client = %Client{},
        computation = %Computation{output_shape: output_shape},
        argument_shapes,
        options \\ []
      ) do
    device_ordinal = Keyword.get(options, :device_ordinal, -1)
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)
    shape_refs = Enum.map(argument_shapes, & &1.ref)
    device_ordinal = check_device_compatibility!(client, device_ordinal)

    # Executable Build Context
    # TODO: Validate replicas, partitions, and shapes

    ref =
      Exla.NIF.compile(
        client.ref,
        computation.ref,
        shape_refs,
        device_ordinal,
        num_replicas,
        num_partitions
      )
      |> unwrap!

    %Executable{
      client: client,
      ref: ref,
      output_shape: output_shape,
      device_ordinal: device_ordinal
    }
  end

  def check_device_compatibility!(client = %Client{}, ordinal) do
    cond do
      ordinal < 0 ->
        Client.get_default_device_ordinal(client)

      ordinal < Client.get_device_count(client) ->
        ordinal

      true ->
        raise ArgumentError, "Invalid device ordinal."
    end
  end
end
