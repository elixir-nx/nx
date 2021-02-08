defmodule EXLA.Device do
  @moduledoc """
  The EXLA implementation of `Nx.Device`.
  """

  @behaviour Nx.Device

  @impl true
  def allocate(data, type, shape, opts) do
    client = opts[:client] || :default
    device_ordinal = opts[:device_ordinal] || -1

    buffer = EXLA.Buffer.buffer(data, EXLA.Shape.make_shape(type, shape))
    buffer = EXLA.Buffer.place_on_device(buffer, EXLA.Client.fetch!(client), device_ordinal)
    {__MODULE__, buffer.ref}
  end

  @impl true
  defdelegate read(data), to: EXLA.Buffer

  @impl true
  defdelegate deallocate(data), to: EXLA.Buffer
end
