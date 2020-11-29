defmodule BufferTest do
  use ExUnit.Case, async: true

  alias Exla.Buffer
  alias Exla.Shape

  import ExlaHelpers

  test "place_on_device/3" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    b2 = Buffer.buffer({<<1::32>>, <<2::32>>}, Shape.make_shape([Shape.make_shape({:s, 32}, {}), Shape.make_shape({:s, 32}, {})]))

    b3_shape = Shape.make_shape([Shape.make_shape([Shape.make_shape({:s, 32}, {})]), Shape.make_shape({:s, 32}, {})])
    b3 = Buffer.buffer({{<<1::32>>}, <<2::32>>}, b3_shape)

    platform = client().platform
    assert %Buffer{ref: {ref, result_platform, 0}} = Buffer.place_on_device(client(), b1, {platform, 0})
    assert result_platform == platform
    assert is_reference(ref)

    assert %Buffer{ref: {ref, result_platform, 0}} = Buffer.place_on_device(client(), b2, {platform, 0})
    assert result_platform == platform
    assert is_reference(ref)

    assert %Buffer{ref: {ref, result_platform, 0}} = Buffer.place_on_device(client(), b3, {platform, 0})
    assert result_platform == platform
    assert is_reference(ref)
  end

  test "read/2" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    b1 = Buffer.place_on_device(client(), b1, {client().platform, 0})

    b2 = Buffer.buffer({<<1::32>>, <<2::32>>}, Shape.make_shape([Shape.make_shape({:s, 32}, {}), Shape.make_shape({:s, 32}, {})]))
    b2 = Buffer.place_on_device(client(), b2, {client().platform, 0})

    b3_shape = Shape.make_shape([Shape.make_shape([Shape.make_shape({:s, 32}, {})]), Shape.make_shape({:s, 32}, {})])
    b3 = Buffer.buffer({{<<1::32>>}, <<2::32>>}, b3_shape)
    b3 = Buffer.place_on_device(client(), b3, {client().platform, 0})

    assert <<1::32>> == Buffer.read(client(), b1)
    # non-destructive
    assert <<1::32>> == Buffer.read(client(), b1)

    assert {<<1::32>>, <<2::32>>} == Buffer.read(client(), b2)
    # non-destructive
    assert {<<1::32>>, <<2::32>>} == Buffer.read(client(), b2)

    assert {{<<1::32>>}, <<2::32>>} == Buffer.read(client(), b3)
    assert {{<<1::32>>}, <<2::32>>} == Buffer.read(client(), b3)

    :ok = Buffer.deallocate(b1)
    assert_raise RuntimeError, "Attempt to read from empty buffer.", fn ->
      Buffer.read(client(), b1)
    end
  end

  test "deallocate/1" do
    b1 = Buffer.buffer(<<1::32>>, Shape.make_shape({:s, 32}, {}))
    b1 = Buffer.place_on_device(client(), b1, {client().platform(), 0})

    b2 = Buffer.buffer({<<1::32>>, <<2::32>>}, Shape.make_shape([Shape.make_shape({:s, 32}, {}), Shape.make_shape({:s, 32}, {})]))
    b2 = Buffer.place_on_device(client(), b2, {client().platform, 0})

    assert :ok = Buffer.deallocate(b1)
    assert :already_deallocated = Buffer.deallocate(b1)

    assert :ok = Buffer.deallocate(b2)
    assert :already_deallocated = Buffer.deallocate(b2)
  end
end
