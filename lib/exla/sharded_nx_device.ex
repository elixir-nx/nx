defmodule EXLA.ShardedNxDevice do
  @moduledoc """
  Implementation of a sharded buffer for pmap.
  """

  @behaviour Nx.Device

  @impl true
  def allocate(data, type, shape, opts) do
    client = opts[:client] || :default

    buffer = EXLA.ShardedBuffer.sharded_buffer(data, EXLA.Shape.make_shape(type, shape))
    buffer = EXLA.ShardedBuffer.place_on_device(buffer, EXLA.Client.fetch!(client))
    {__MODULE__, buffer.buffers}
  end

  @impl true
  defdelegate read(data), to: EXLA.ShardedBuffer

  @impl true
  defdelegate deallocate(data), to: EXLA.ShardedBuffer
end
