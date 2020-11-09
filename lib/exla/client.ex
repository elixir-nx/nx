defmodule Exla.Client do
  alias Exla.Options.LocalClientOptions

  @moduledoc """
  Wrapper around various `xla::Client`, `xla::LocalClient`, and `xla::ClientLibrary` methods.
  """

  # It is possible, although I don't think it's preferable, to create several instances of different clients.
  # The Python XLA Extension wraps a `Client` class with methods for creating CPU, GPU, and TPU clients...
  # We could probably mirror this functionality, although it may be unnecessary in most cases to force a
  # user to create a client for every application. We could probably wrap this in a GenServer with a supervisor
  # and maintain client references that way.
  def create(options = %LocalClientOptions{}) do
    case Exla.NIF.get_or_create_local_client(options) do
      :ok -> :ok
      _   -> {:error, "Unable to create client."}
    end
  end
end
