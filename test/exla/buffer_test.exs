defmodule BufferTest do
  use ExUnit.Case, async: true

  alias Exla.Buffer
  alias Exla.Shape

  import ExlaHelpers

  test "place_on_device/3" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    assert %Buffer{ref: {ref, :default}} = Buffer.place_on_device(b1, client(), 0)
    assert is_reference(ref)
  end

  test "read/2" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    b1 = Buffer.place_on_device(b1, client(), 0)

    assert <<1::32>> == Buffer.read(b1.ref)
    # non-destructive
    assert <<1::32>> == Buffer.read(b1.ref)

    :ok = Buffer.deallocate(b1.ref)

    assert_raise RuntimeError, "Attempt to read from empty buffer.", fn ->
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
