defmodule Exla.Client do
  alias __MODULE__, as: Client
  alias Exla.Options.LocalClientOptions
  alias Exla.Options.ExecutableBuildOptions
  alias Exla.Computation
  alias Exla.LocalExecutable
  @enforce_keys [:ref]
  defstruct [:ref]

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
    {:ok, ref} = Exla.NIF.compile(client, computation, argument_shapes, options)
    %LocalExecutable{ref: ref}
  end
end
