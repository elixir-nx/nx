defmodule EXLA.ExecutableTest do
  use ExUnit.Case, async: true

  alias EXLA.{BinaryBuffer, DeviceBuffer, Executable, Op, Shape}
  import EXLAHelpers

  test "raises on invalid tuples" do
    t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
    t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

    assert_raise ArgumentError, ~r"can only compile computations with a tuple at the root", fn ->
      run_one([t1, t2], [], fn b, x, y ->
        Op.tuple(b, [Op.tuple(b, [x]), Op.tuple(b, [y])])
      end)
    end

    assert_raise ArgumentError, ~r"can only compile computations with a tuple at the root", fn ->
      run_one([t1, t2], [], fn _b, x, y -> Op.add(x, y) end)
    end
  end

  describe "run" do
    test "with no inputs and default options" do
      assert [a = %DeviceBuffer{}] =
               run_one([], fn b ->
                 Op.tuple(b, [Op.constant_r0(b, 1, {:s, 32})])
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
    end

    test "with 2 inputs and default options" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}] =
               run_one([t1, t2], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)

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
               run_one([t1, t2], [], fn b, x, y ->
                 Op.tuple(b, [Op.add(x, y)])
               end)

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [], fn b, x, y ->
                 Op.tuple(b, [Op.add(x, y)])
               end)

      assert DeviceBuffer.read(t1) == <<1::32-native>>
      assert DeviceBuffer.read(t2) == <<1::32-native>>
    end

    test "with data from a previous run" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      exec = compile([t1.shape, t2.shape], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
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
               run_one([t1, t2], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)

      assert <<3::32-native>> == DeviceBuffer.read(a)
    end

    test "with tuple return" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}] =
               run_one([t1, t2], fn b, x, y -> Op.tuple(b, [x, y]) end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert <<2::32-native>> == DeviceBuffer.read(b)
    end

    @tag :multi_device
    test "runs on a specific device" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}, c = %DeviceBuffer{}] =
               run_one([t1, t2], [device_id: 1], fn b, x, y ->
                 Op.tuple(b, [x, y, Op.add(x, y)])
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert a.device_id == 1
      assert <<2::32-native>> == DeviceBuffer.read(b)
      assert b.device_id == 1
      assert <<3::32-native>> == DeviceBuffer.read(c)
      assert c.device_id == 1

      assert_raise RuntimeError, ~r"Expected buffer to be placed on device 0", fn ->
        run_one([a, b], [device_id: 0], fn b, x, y ->
          Op.tuple(b, [Op.add(x, y)])
        end)
      end
    end
  end
end

defmodule EXLA.ExecutableFeedTest do
  # infeed/outfeed are global resources, so they either
  # need to be locked or we cannot run them concurrently.
  use ExUnit.Case, async: false

  alias EXLA.{BinaryBuffer, DeviceBuffer, Client, Op, Shape}
  import EXLAHelpers

  describe "infeed/outfeed" do
    test "successfully sends to/from device asynchronously" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert res =
               Task.async(fn ->
                 run_one([], [], fn b ->
                   token = Op.create_token(b)
                   val_and_token = Op.infeed(token, t.shape)
                   val = Op.get_tuple_element(val_and_token, 0)
                   new_token = Op.get_tuple_element(val_and_token, 1)
                   outfeed_val = Op.add(val, val)
                   _outfeed_token = Op.outfeed(outfeed_val, new_token)
                   Op.tuple(b, [Op.add(outfeed_val, val)])
                 end)
               end)

      assert :ok = Client.to_infeed(client(), 0, [{t.data, t.shape}])
      assert from_outfeed(client(), 0, Shape.make_shape({:s, 32}, {})) == <<2::32-native>>

      assert [a = %DeviceBuffer{}] = Task.await(res)
      assert DeviceBuffer.read(a) == <<3::32-native>>
    end

    test "successfully sends to/from device asynchronously in a loop" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert res =
               Task.async(fn ->
                 run_one([], [], fn b ->
                   token_shape = Shape.make_token_shape()
                   tuple_shape = Shape.make_tuple_shape([t.shape, token_shape])

                   condition_b = EXLA.Builder.new(b, "condition")
                   param = EXLA.Op.parameter(condition_b, 0, tuple_shape, "arg")
                   zero = Op.constant_r0(condition_b, 0, {:s, 32})
                   val = Op.get_tuple_element(param, 0)
                   condition = EXLA.Builder.build(Op.not_equal(val, zero))

                   while_b = EXLA.Builder.new(b, "while")
                   param = EXLA.Op.parameter(while_b, 0, tuple_shape, "arg")
                   val = Op.get_tuple_element(param, 0)
                   token = Op.get_tuple_element(param, 1)
                   token = Op.outfeed(Op.add(val, val), token)
                   while = EXLA.Builder.build(Op.infeed(token, t.shape))

                   token = Op.create_token(b)
                   while = Op.while(condition, while, Op.infeed(token, t.shape))
                   Op.tuple(b, [Op.get_tuple_element(while, 0)])
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
