defmodule Exla.Client do
  alias __MODULE__, as: Client
  alias Exla.Computation
  alias Exla.Executable
  @enforce_keys [:ref, :platform]
  defstruct [:ref, :platform]

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
        client = %Client{platform: platform},
        computation = %Computation{},
        argument_shapes,
        options \\ []
      ) do
    case platform do
      :cuda -> _compile_cuda(client, computation, argument_shapes, options)
      :host -> _compile_host(client, computation, argument_shapes, options)
    end
  end

  defp _compile_cuda(
         client = %Client{platform: :cuda},
         computation = %Computation{},
         argument_shapes,
         options
       ) do
    # TODO: We need this in order for the `Compile` NIF to start `ptxas` using a TF Subprocess. The subprocess relies on `waitpid`
    # which fails under normal circumstances because ERTS sets SIGCHLD to SIGIGN. We need to determine the implications of setting
    # this here.
    :os.set_signal(:sigchld, :default)
    _compile(client, computation, argument_shapes, options)
  end

  defp _compile_host(
         client = %Client{platform: :host},
         computation = %Computation{},
         argument_shapes,
         options
       ) do
    _compile(client, computation, argument_shapes, options)
  end

  defp _compile(
         client = %Client{ref: ref},
         %Computation{ref: computation},
         argument_shapes,
         options
       ) do
    device_ordinal = Keyword.get(options, :device_ordinal, -1)
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)
    # TODO: I think argument shapes should be a list since we have to traverse it to pull out
    # the refs of each Shape. To simplify the handling of `absl::Span` on the NIF side
    # I only read spans in as Tuples. This is important because things like the dimensions
    # of a shape, broadcast dimensions, etc. naturally fit well with Tuples, but other things
    # that use spans such as this argument here work better with lists. I suppose I could create
    # two distinct methods for handling lists and tuples.
    shape_refs =
      argument_shapes
      |> Tuple.to_list()
      |> Enum.map(& &1.ref)
      |> List.to_tuple()

    # Executable Build Context
    # TODO: Validate replicas, partitions, and shapes
    with {:ok, {_platform, device_ordinal}} <-
           check_device_compatibility(client, {client.platform, device_ordinal}),
         {:ok, ref} <-
           Exla.NIF.compile(
             ref,
             computation,
             shape_refs,
             device_ordinal,
             num_replicas,
             num_partitions
           ) do
      %Executable{client: client, ref: ref, device: {client.platform, device_ordinal}}
    end
  end

  def check_device_compatibility(
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
