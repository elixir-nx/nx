defmodule Exla.Client do
  alias __MODULE__, as: Client
  alias Exla.Options.LocalClientOptions
  alias Exla.Options.ExecutableBuildOptions
  alias Exla.Computation
  alias Exla.LocalExecutable
  @enforce_keys [:ref]
  defstruct [:ref]

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
  def create_client(options \\ %LocalClientOptions{}) do
    # TODO: Rename this function to get_local_client. It is a singleton,
    # non-thread-safe resource in XLA so we need to mimic the same
    # in Elixir. We should also have distinct steps for configuring and for
    # getting it. See: https://github.com/seanmor5/exla/pull/12
    {:ok, ref} =
      Exla.NIF.get_or_create_local_client(
        options.platform,
        options.number_of_replicas,
        options.intra_op_parallelism_threads
      )

    %Client{ref: ref}
  end

  def get_device_count(%Client{ref: client}) do
    Exla.NIF.get_device_count(client)
  end

  def compile(
        %Client{ref: client},
        %Computation{ref: computation},
        argument_shapes,
        options \\ %ExecutableBuildOptions{}
  ) do
    # TODO: I think argument shapes should be a list since we have to traverse it to pull out
    # the refs of each Shape. To simplify the handling of `absl::Span` on the NIF side
    # I only read spans in as Tuples. This is important because things like the dimensions
    # of a shape, broadcast dimensions, etc. naturally fit well with Tuples, but other things
    # that use spans such as this argument here work better with lists. I suppose I could create
    # two distinct methods for handling lists and tuples.
    shape_refs =
      argument_shapes
      |> Tuple.to_list()
      |> Enum.map(&(&1.ref))
      |> List.to_tuple()
    {:ok, ref} = Exla.NIF.compile(client, computation, shape_refs, options)
    %LocalExecutable{ref: ref}
  end
end
