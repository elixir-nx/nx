defmodule Exla.ShardedNxDevice do
  @moduledoc """
  Implementation of a sharded buffer for pmap.
  """

  @behaviour Nx.Device

  @impl true
  def allocate(data, type, shape, opts) do
    client = opts[:client] || :default

    buffer = Exla.ShardedBuffer.sharded_buffer(data, Exla.Shape.make_shape(type, shape))
    buffer = Exla.ShardedBuffer.place_on_device(buffer, Exla.Client.fetch!(client))
    {__MODULE__, buffer.buffers}
  end

  @impl true
  defdelegate read(data), to: Exla.ShardedBuffer

  @impl true
  defdelegate deallocate(data), to: Exla.ShardedBuffer
end