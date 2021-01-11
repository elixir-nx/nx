defmodule EXLA.ShardedBuffer do
  @moduledoc """
  A sharded buffer.
  """

  alias __MODULE__
  alias EXLA.{Buffer, Client, Shape}

  @enforce_keys [:shape]
  defstruct [:buffers, :shape]

  @doc """
  Creates a new sharded buffer
  """
  def sharded_buffer(binary_or_reference_pairs, shape)

  def sharded_buffer([reference_pair | []], shape = %Shape{dims: {}}) do
    buffers = [Buffer.buffer(reference_pair, shape)]
    %ShardedBuffer{buffers: buffers, shape: shape}
  end

  def sharded_buffer(binary, shape = %Shape{dtype: {_, size} = type, dims: dims})
      when is_binary(binary) do
    num_shards = elem(dims, 0)
    sharded_dims = Tuple.delete_at(dims, 0)
    sharded_shape = Shape.make_shape(type, sharded_dims)
    shard_size = tuple_product(sharded_dims) * size

    buffers =
      for i <- 0..(num_shards - 1) do
        consumed = i * shard_size
        <<_::size(consumed)-bitstring, shard::size(shard_size)-bitstring, _::bitstring>> = binary
        Buffer.buffer(shard, sharded_shape)
      end

    %ShardedBuffer{buffers: buffers, shape: shape}
  end

  def sharded_buffer(reference_pairs, shape = %Shape{dtype: type, dims: dims})
      when is_list(reference_pairs) do
    num_shards = elem(dims, 0)
    num_buffers = Enum.count(reference_pairs)
    sharded_shape = Shape.make_shape(type, Tuple.delete_at(dims, 0))

    unless num_shards == num_buffers,
      do:
        raise(
          "expected input number of buffers to match number of shards," <>
            " got #{num_shards} shards and #{num_buffers} buffers"
        )

    buffers =
      reference_pairs
      |> Enum.map(&Buffer.buffer(&1, sharded_shape))

    %ShardedBuffer{buffers: buffers, shape: shape}
  end

  @doc """
  Places the sharded buffer on devices.
  """
  def place_on_device(
        sharded = %ShardedBuffer{buffers: buffers},
        client = %Client{device_count: device_count}
      ) do
    num_shards = Enum.count(buffers)

    unless num_shards <= device_count,
      do:
        raise(
          "expected size of sharding axis to be less than or equal to" <>
            " the number of available devices on client. Axis size is" <>
            " #{num_shards}, device count is #{device_count}"
        )

    ref_buffers =
      buffers
      |> Enum.with_index()
      |> Enum.map(fn {buf, i} -> Buffer.place_on_device(buf, client, i) end)

    %ShardedBuffer{sharded | buffers: ref_buffers}
  end

  @doc """
  Reads the underlying shards.
  """
  def read(shards) do
    for shard <- shards, into: <<>> do
      Buffer.read(shard.ref)
    end
  end

  @doc """
  Deallocates the underlying shards.
  """
  def deallocate(shards) do
    shards
    |> Enum.reduce(:ok, fn x, _ -> Buffer.deallocate(x.ref) end)
  end

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)
end
