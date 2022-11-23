defmodule EXLA.MultiDeviceTest do
  use ExUnit.Case, async: true

  alias EXLA.{BinaryBuffer, DeviceBuffer, Executable, Op, Shape}
  import EXLAHelpers

  @moduletag :multi_device

  test "succeeds with device set" do
    t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
    t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

    assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}, c = %DeviceBuffer{}] =
             run_one([t1, t2], [device_id: 1], fn b, x, y ->
               Op.tuple(b, [x, y, Op.add(x, y)])
             end)

    assert <<1::32-native>> == DeviceBuffer.read(a)
    assert <<2::32-native>> == DeviceBuffer.read(b)
    assert <<3::32-native>> == DeviceBuffer.read(c)
  end

  @tag :skip
  test "succeeds with num_replicas > 1" do
    t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
    t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))
    t3 = BinaryBuffer.from_binary(<<3::32-native>>, Shape.make_shape({:s, 32}, {}))
    t4 = BinaryBuffer.from_binary(<<4::32-native>>, Shape.make_shape({:s, 32}, {}))

    executable =
      compile([t1.shape, t2.shape], [num_replicas: 2], fn b, x, y ->
        Op.tuple(b, [x, y, Op.add(x, y)])
      end)

    assert executable.device_id == -1

    [[a1, a2, a3], [b1, b2, b3]] = Executable.run(executable, [[t1, t2], [t3, t4]])

    assert <<1::32-native>> == DeviceBuffer.read(a1)
    assert <<2::32-native>> == DeviceBuffer.read(a2)
    assert <<3::32-native>> == DeviceBuffer.read(a3)
    assert a1.device_id == a2.device_id
    assert a2.device_id == a3.device_id

    assert <<3::32-native>> == DeviceBuffer.read(b1)
    assert <<4::32-native>> == DeviceBuffer.read(b2)
    assert <<7::32-native>> == DeviceBuffer.read(b3)
    assert b1.device_id == b2.device_id
    assert b2.device_id == b3.device_id

    assert a1.device_id != b1.device_id
  end
end
