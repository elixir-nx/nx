defmodule Exla.Client do
  alias __MODULE__, as: Client
  alias Exla.Options.LocalClientOptions

  @enforce_keys [:ref]
  defstruct [:ref]

  def create_client(options \\ %LocalClientOptions{}) do
    # TODO: Rename this function to get_local_client. It is a singleton,
    # non-thread-safe resource in XLA so we need to mimic the same
    # in Elixir. We should also have distinct steps for configuring and for
    # getting it. See: https://github.com/seanmor5/exla/pull/12
    case Exla.NIF.get_or_create_local_client(
           options.platform,
           options.number_of_replicas,
           options.intra_op_parallelism_threads
         ) do
      {:ok, ref} -> {:ok, %Client{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  def get_device_count(client = %Client{}) do
    Exla.NIF.get_device_count(client.ref)
  end
end
