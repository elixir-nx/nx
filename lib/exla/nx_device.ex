defmodule Exla.NxDevice do
  @moduledoc """
  The Exla implementation of `Nx.Device`.
  """

  @behaviour Nx.Device

  @impl true
  def allocate(data, type, shape, opts) do
    client = opts[:client] || :default
    device_ordinal = opts[:device_ordinal] || -1

    buffer = Exla.Buffer.buffer(data, Exla.Shape.make_shape(type, shape))
    buffer = Exla.Buffer.place_on_device(buffer, Exla.Client.fetch!(client), device_ordinal)
    {__MODULE__, buffer.ref}
  end

  @impl true
  defdelegate read(data), to: Exla.Buffer

  @impl true
  defdelegate deallocate(data), to: Exla.Buffer
end