defmodule EXLA.BufferTest do
  use ExUnit.Case, async: true

  alias EXLA.{Buffer, Shape, ShardedBuffer}

  import EXLAHelpers

  describe "buffer" do
    test "place_on_device/3" do
      b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
      assert %Buffer{ref: {ref, :default}} = Buffer.place_on_device(b1, client(), 0)
      assert is_reference(ref)
    end

    test "read/2" do
      b1 = Buffer.buffer(<<1::32, 2::32, 3::32, 4::32>>, Shape.make_shape({:s, 32}, {4}))
      b1 = Buffer.place_on_device(b1, client(), 0)

      # non-destructive
      assert <<1::32, 2::32, 3::32, 4::32>> == Buffer.read(b1.ref)
      assert <<1::32, 2::32, 3::32, 4::32>> == Buffer.read(b1.ref)

      # doesn't change after deallocation
      binary = Buffer.read(b1.ref)
      :ok = Buffer.deallocate(b1.ref)
      assert binary == <<1::32, 2::32, 3::32, 4::32>>

      assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn ->
        Buffer.read(b1.ref)
      end
    end

    test "deallocate/1" do
      b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
      b1 = Buffer.place_on_device(b1, client(), 0)

      assert :ok = Buffer.deallocate(b1.ref)
      assert :already_deallocated = Buffer.deallocate(b1.ref)
    end
  end

  describe "sharded buffer" do
    @describetag :multi_device

    test "sharded_buffer/2" do
      data = for i <- 1..2000, into: <<>>, do: <<i::64-native>>
      shape = Shape.make_shape({:s, 64}, {2, 1000})

      assert %ShardedBuffer{buffers: buffers} = ShardedBuffer.sharded_buffer(data, shape)
      assert is_list(buffers)
    end

    test "place_on_device/3" do
      data = for i <- 1..2000, into: <<>>, do: <<i::64-native>>
      shape = Shape.make_shape({:s, 64}, {2, 1000})
      b1 = ShardedBuffer.sharded_buffer(data, shape)

      assert %ShardedBuffer{buffers: buffers} = ShardedBuffer.place_on_device(b1, client())

      for buf <- buffers do
        assert %Buffer{ref: {ref, _}} = buf
        assert is_reference(ref)
      end
    end

    test "read/2" do
      data = for i <- 1..4, into: <<>>, do: <<i::64>>
      shape = Shape.make_shape({:s, 64}, {2, 2})
      b1 = ShardedBuffer.sharded_buffer(data, shape)
      b1 = ShardedBuffer.place_on_device(b1, client())

      # non-destructive
      assert <<1::64, 2::64, 3::64, 4::64>> == ShardedBuffer.read(b1.buffers)
      assert <<1::64, 2::64, 3::64, 4::64>> == ShardedBuffer.read(b1.buffers)

      # doesn't change after deallocation
      binary = ShardedBuffer.read(b1.buffers)
      :ok = ShardedBuffer.deallocate(b1.buffers)
      assert binary == <<1::64, 2::64, 3::64, 4::64>>

      assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn ->
        ShardedBuffer.read(b1.buffers)
      end
    end

    test "deallocate/1" do
      data = for i <- 1..4, into: <<>>, do: <<i::64>>
      shape = Shape.make_shape({:s, 64}, {2, 2})
      b1 = ShardedBuffer.sharded_buffer(data, shape)
      b1 = ShardedBuffer.place_on_device(b1, client())

      assert :ok = ShardedBuffer.deallocate(b1.buffers)
      assert :already_deallocated = ShardedBuffer.deallocate(b1.buffers)
    end
  end
end
