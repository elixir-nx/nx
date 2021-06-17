defmodule EXLA.BufferTest do
  use ExUnit.Case, async: true

  alias EXLA.{Buffer, Shape}

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

      assert_raise RuntimeError, "CopyToHostAsync() called on deleted or donated buffer", fn ->
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
end
