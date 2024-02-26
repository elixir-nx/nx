defmodule EXLA.ExecutableTest do
  use ExUnit.Case, async: true

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Executable
  alias EXLA.Op
  alias EXLA.Shape
  alias EXLA.MLIR.Value
  import EXLAHelpers

  describe "run" do
    test "with no inputs and default options" do
      assert [a = %DeviceBuffer{}] =
               run_one([], [], Shape.make_shape({:s, 32}, {}), fn b ->
                 mod().tuple(b, [mod().constant_r0(b, 1, {:s, 32})])
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
    end

    test "with 2 inputs and default options" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}] =
               run_one([t1, t2], [], Shape.make_tuple_shape([t1.shape]), fn b, x, y ->
                 mod().tuple(b, [add(x, y)])
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
                 mod().tuple(b, [add(x, y)])
               end)

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [], Shape.make_tuple_shape([t1.shape]), fn b, x, y ->
                 mod().tuple(b, [add(x, y)])
               end)

      assert DeviceBuffer.read(t1) == <<1::32-native>>
      assert DeviceBuffer.read(t2) == <<1::32-native>>
    end

    test "with data from a previous run" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      exec =
        compile([t1.shape, t2.shape], [], [t1.shape], fn b, x, y ->
          mod().tuple(b, [add(x, y)])
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
               run_one([t1, t2], [], {t1.shape}, fn b, x, y -> mod().tuple(b, [add(x, y)]) end)

      assert <<3::32-native>> == DeviceBuffer.read(a)
    end

    test "with tuple return" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}] =
               run_one([t1, t2], [], Shape.make_tuple_shape([t1.shape, t2.shape]), fn b, x, y ->
                 mod().tuple(b, [x, y])
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
                   mod().tuple(b, [x, y, add(x, y)])
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
          mod().tuple(b, [add(x, y)])
        end)
      end
    end
  end

  defp add(x, y) do
    if mod() == Value do
      Value.add(x.function, x, y)
    else
      Op.add(x, y)
    end
  end

  defp mod do
    if Application.get_env(:exla, :compiler_mode) == :mlir do
      Value
    else
      Op
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
  alias EXLA.Op
  alias EXLA.Shape
  alias EXLA.MLIR.Value
  import EXLAHelpers

  defp mod do
    if Application.get_env(:exla, :compiler_mode) == :mlir do
      Value
    else
      Op
    end
  end

  describe "infeed/outfeed" do
    test "successfully sends to/from device asynchronously" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert res =
               Task.async(fn ->
                 run_one([], [], Shape.make_tuple_shape([Shape.make_token_shape()]), fn b ->
                   token = mod().create_token(b)

                   {new_token, val} =
                     if mod() == Value do
                       {new_token, [val]} = Value.infeed(token, t.shape)
                       {new_token, val}
                     else
                       val_and_token = Op.infeed(token, t.shape)
                       val = Op.get_tuple_element(val_and_token, 1)
                       new_token = Op.get_tuple_element(val_and_token, 0)
                       {new_token, val}
                     end

                   outfeed_val = add(val, val)
                   _outfeed_token = mod().outfeed(outfeed_val, new_token)
                   mod().tuple(b, [add(outfeed_val, val)])
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
                   if mod() == Value do
                     condition =
                       EXLA.MLIR.Module.add_function(
                         b.module,
                         "condition",
                         [token_shape, t.shape],
                         [
                           token_shape,
                           Shape.make_shape({:pred, 8}, {})
                         ]
                       )

                     [_token, val] = EXLA.MLIR.Function.get_arguments(condition)
                     zero = Value.constant_r0(condition, 0, {:s, 32})
                     Value.variadic_return([Value.not_equal(condition, val, zero)])

                     body =
                       EXLA.MLIR.Module.add_function(b.module, "body", [token_shape, t.shape], [
                         token_shape,
                         t.shape
                       ])

                     [token, val] = EXLA.MLIR.Function.get_arguments(body)

                     token = Value.outfeed(Value.add(body, val, val), token)

                     {token, [input]} = Value.infeed(token, t.shape)

                     Value.variadic_return([token, input])

                     token = Value.create_token(b)
                     {token, [input]} = Value.infeed(token, t.shape)

                     [_token, result] = Value.while(condition, body, [token, input])
                     Value.tuple(b, [result])
                   else
                     tuple_shape = Shape.make_tuple_shape([t.shape, Shape.make_token_shape()])
                     condition_b = EXLA.Builder.new(b, "condition")
                     param = mod().parameter(condition_b, 0, tuple_shape, "arg")
                     zero = mod().constant_r0(condition_b, 0, {:s, 32})
                     val = mod().get_tuple_element(param, 0)
                     condition = EXLA.Builder.build(mod().not_equal(val, zero))

                     while_b = EXLA.Builder.new(b, "while")
                     param = mod().parameter(while_b, 0, tuple_shape, "arg")
                     val = mod().get_tuple_element(param, 0)
                     token = mod().get_tuple_element(param, 1)
                     token = mod().outfeed(add(val, val), token)
                     while = EXLA.Builder.build(mod().infeed(token, t.shape))

                     token = mod().create_token(b)
                     while = mod().while(condition, while, mod().infeed(token, t.shape))
                     mod().tuple(b, [mod().get_tuple_element(while, 0)])
                   end
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

  defp add(x, y) do
    if mod() == Value do
      Value.add(x.function, x, y)
    else
      Op.add(x, y)
    end
  end
end
