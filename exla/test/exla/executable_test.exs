defmodule EXLA.ExecutableTest do
  use ExUnit.Case, async: true

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Executable
  alias EXLA.Shape
  alias EXLA.MLIR.Value
  import EXLAHelpers

  describe "run" do
    test "with no inputs and default options" do
      assert [a = %DeviceBuffer{}] =
               run_one([], [], Shape.make_shape({:s, 32}, {}), fn b ->
                 Value.tuple(b, [Value.constant_r0(b, 1, {:s, 32})])
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
    end

    test "with 2 inputs and default options" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}] =
               run_one([t1, t2], [], Shape.make_tuple_shape([t1.shape]), fn b, x, y ->
                 Value.tuple(b, [Value.add(b, x, y)])
               end)

      assert <<2::32-native>> == DeviceBuffer.read(a)
    end

    test "when data is preloaded" do
      t1 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          Shape.make_shape({:s, 32}, {}),
          client(),
          0
        )

      t2 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          Shape.make_shape({:s, 32}, {}),
          client(),
          0
        )

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [], t1.shape, fn b, x, y ->
                 Value.tuple(b, [Value.add(b, x, y)])
               end)

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [], Shape.make_tuple_shape([t1.shape]), fn b, x, y ->
                 Value.tuple(b, [Value.add(b, x, y)])
               end)

      assert DeviceBuffer.read(t1) == <<1::32-native>>
      assert DeviceBuffer.read(t2) == <<1::32-native>>
    end

    test "with data from a previous run" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      exec =
        compile([t1.shape, t2.shape], [], [t1.shape], fn b, x, y ->
          Value.tuple(b, [Value.add(b, x, y)])
        end)

      assert [[t3 = %DeviceBuffer{}]] = Executable.run(exec, [[t1, t2]])
      assert [[a = %DeviceBuffer{}]] = Executable.run(exec, [[t3, t3]])

      assert <<4::32-native>> == DeviceBuffer.read(a)
    end

    test "with mixed data" do
      t1 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          Shape.make_shape({:s, 32}, {}),
          client(),
          0
        )

      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}] =
               run_one([t1, t2], [], {t1.shape}, fn b, x, y ->
                 Value.tuple(b, [Value.add(b, x, y)])
               end)

      assert <<3::32-native>> == DeviceBuffer.read(a)
    end

    test "with tuple return" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}] =
               run_one([t1, t2], [], Shape.make_tuple_shape([t1.shape, t2.shape]), fn b, x, y ->
                 Value.tuple(b, [x, y])
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert <<2::32-native>> == DeviceBuffer.read(b)
    end

    @tag :multi_device
    test "runs on a specific device" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}, c = %DeviceBuffer{}] =
               run_one(
                 [t1, t2],
                 [device_id: 1],
                 EXLA.Shape.make_tuple_shape([t1.shape, t2.shape, t1.shape]),
                 fn b, x, y ->
                   Value.tuple(b, [x, y, Value.add(b, x, y)])
                 end
               )

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert a.device_id == 1
      assert <<2::32-native>> == DeviceBuffer.read(b)
      assert b.device_id == 1
      assert <<3::32-native>> == DeviceBuffer.read(c)
      assert c.device_id == 1

      assert_raise RuntimeError, ~r"Expected buffer to be placed on device 0", fn ->
        run_one([a, b], [device_id: 0], t1.shape, fn b, x, y ->
          Value.tuple(b, [Value.add(b, x, y)])
        end)
      end
    end
  end
end

defmodule EXLA.ExecutableFeedTest do
  # infeed/outfeed are global resources, so they either
  # need to be locked or we cannot run them concurrently.
  use ExUnit.Case, async: false

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Client
  alias EXLA.Shape
  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value
  import EXLAHelpers

  describe "infeed/outfeed" do
    test "successfully sends to/from device asynchronously" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert res =
               Task.async(fn ->
                 run_one([], [], Shape.make_tuple_shape([Shape.make_token_shape()]), fn b ->
                   token = Value.create_token(b)

                   {new_token, [val]} = Value.infeed(token, t.shape)

                   outfeed_val = Value.add(b, val, val)
                   _outfeed_token = Value.outfeed(outfeed_val, new_token)
                   Value.tuple(b, [Value.add(b, outfeed_val, val)])
                 end)
               end)

      assert :ok = Client.to_infeed(client(), 0, [{t.data, t.shape}])
      assert from_outfeed(client(), 0, Shape.make_shape({:s, 32}, {})) == <<2::32-native>>

      assert [a = %DeviceBuffer{}] = Task.await(res)
      assert DeviceBuffer.read(a) == <<3::32-native>>
    end

    test "successfully sends to/from device asynchronously in a loop" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      token_shape = Shape.make_token_shape()

      assert res =
               Task.async(fn ->
                 run_one([], [], {token_shape, t.shape}, fn b ->
                   token = Value.create_token(b)

                   {token, [val]} = Value.infeed(token, t.shape)

                   {[_token, result], condition_region, body_region} =
                     Value.while(b, [token, val])

                   [_token, val] = Function.push_region(b, condition_region)
                   zero = Value.constant_r0(b, 0, {:s, 32})
                   Value.variadic_return([Value.not_equal(b, val, zero)])
                   Function.pop_region(b)

                   [body_token, val] = Function.push_region(b, body_region)

                   body_token = Value.outfeed(Value.add(b, val, val), body_token)
                   {body_token, [input]} = Value.infeed(body_token, t.shape)

                   Value.variadic_return([body_token, input])
                   Function.pop_region(b)

                   Value.tuple(b, [result])
                 end)
               end)

      assert :ok = Client.to_infeed(client(), 0, [{<<1::32-native>>, t.shape}])
      assert from_outfeed(client(), 0, Shape.make_shape({:s, 32}, {})) == <<2::32-native>>

      assert :ok = Client.to_infeed(client(), 0, [{<<2::32-native>>, t.shape}])
      assert from_outfeed(client(), 0, Shape.make_shape({:s, 32}, {})) == <<4::32-native>>

      assert :ok = Client.to_infeed(client(), 0, [{<<0::32-native>>, t.shape}])

      assert [a = %DeviceBuffer{}] = Task.await(res)
      assert DeviceBuffer.read(a) == <<0::32-native>>
    end
  end

  defp from_outfeed(client, device_id, shape) do
    ref = make_ref()
    Client.from_outfeed(client, device_id, [shape], self(), ref)

    receive do
      {^ref, msg} -> msg
    end
  end
end
