defmodule Exla.Client do
  alias __MODULE__, as: Client
  alias Exla.Options.LocalClientOptions

  @enforce_keys [:ref]
  defstruct [:ref]

  # There's a few scenarios where this could fail, is this the best way to handle failure on startup?
  def create_client(options = %LocalClientOptions{}) do
    case Exla.NIF.get_or_create_local_client(
           options.platform,
           options.number_of_replicas,
           options.intra_op_parallelism_threads
         ) do
      {:ok, ref} -> {:ok, %Client{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end
end
