defmodule BufferTest do
  use ExUnit.Case, async: true

  alias Exla.Buffer
  alias Exla.Shape

  import ExlaHelpers

  test "place_on_device/3" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    assert %Buffer{ref: ref} = Buffer.place_on_device(client(), b1, {client().platform, 0})
    assert is_reference(ref)
  end

  test "read/2" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    b1 = Buffer.place_on_device(client(), b1, {client().platform, 0})
    assert <<1::32>> == Buffer.read(client(), b1)
    # non-destructive
    assert <<1::32>> == Buffer.read(client(), b1)

    :ok = Buffer.deallocate(b1)
    assert_raise RuntimeError, "Attempt to read from empty buffer.", fn ->
      Buffer.read(client(), b1)
    end
  end

  test "deallocate/1" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    assert_raise RuntimeError, "Attempt to deallocate nothing.", fn ->
      Buffer.deallocate(b1)
    end
    b1 = Buffer.place_on_device(client(), b1, {client().platform(), 0})
    assert :ok = Buffer.deallocate(b1)
    assert_raise RuntimeError, "Attempt to deallocate already deallocated buffer.", fn ->
      Buffer.deallocate(b1)
    end
  end
end