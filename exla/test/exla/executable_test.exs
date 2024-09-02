defmodule EXLA.ExecutableTest do
  use ExUnit.Case, async: true

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Executable
  alias EXLA.Typespec
  alias EXLA.MLIR.Value
  import EXLAHelpers

  describe "run" do
    test "with no inputs and default options" do
      assert [a = %DeviceBuffer{}] =
               run_one([], [], s32_typespec(), fn b ->
                 [Value.constant(b, [1], s32_typespec())]
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
    end

    test "with 2 inputs and default options" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())

      assert [a = %DeviceBuffer{}] =
               run_one([t1, t2], [], [t1.typespec], fn _b, x, y ->
                 [Value.add(x, y, s32_typespec())]
               end)

      assert <<2::32-native>> == DeviceBuffer.read(a)
    end

    test "when data is preloaded" do
      t1 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          s32_typespec(),
          client(),
          0
        )

      t2 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          s32_typespec(),
          client(),
          0
        )

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [], t1.typespec, fn _b, x, y ->
                 [Value.add(x, y, s32_typespec())]
               end)

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [], [t1.typespec], fn _b, x, y ->
                 [Value.add(x, y, s32_typespec())]
               end)

      assert DeviceBuffer.read(t1) == <<1::32-native>>
      assert DeviceBuffer.read(t2) == <<1::32-native>>
    end

    test "with data from a previous run" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())

      exec =
        compile([t1.typespec, t2.typespec], [], [t1.typespec], fn _b, x, y ->
          [Value.add(x, y, s32_typespec())]
        end)

      assert [[t3 = %DeviceBuffer{}]] = Executable.run(exec, [[t1, t2]])
      assert [[a = %DeviceBuffer{}]] = Executable.run(exec, [[t3, t3]])

      assert <<4::32-native>> == DeviceBuffer.read(a)
    end

    test "with mixed data" do
      t1 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          s32_typespec(),
          client(),
          0
        )

      t2 = BinaryBuffer.from_binary(<<2::32-native>>, s32_typespec())

      assert [a = %DeviceBuffer{}] =
               run_one([t1, t2], [], [t1.typespec], fn _b, x, y ->
                 [Value.add(x, y, s32_typespec())]
               end)

      assert <<3::32-native>> == DeviceBuffer.read(a)
    end

    test "with tuple return" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, s32_typespec())

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}] =
               run_one([t1, t2], [], [t1.typespec, t2.typespec], fn _b, x, y ->
                 [x, y]
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert <<2::32-native>> == DeviceBuffer.read(b)
    end

    @tag :multi_device
    test "runs on a specific device" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, s32_typespec())

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}, c = %DeviceBuffer{}] =
               run_one(
                 [t1, t2],
                 [device_id: 1],
                 [t1.typespec, t2.typespec, t1.typespec],
                 fn _b, x, y ->
                   [x, y, Value.add(x, y, s32_typespec())]
                 end
               )

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert a.device_id == 1
      assert <<2::32-native>> == DeviceBuffer.read(b)
      assert b.device_id == 1
      assert <<3::32-native>> == DeviceBuffer.read(c)
      assert c.device_id == 1

      assert_raise RuntimeError, ~r"Expected buffer to be placed on device 0", fn ->
        run_one([a, b], [device_id: 0], t1.typespec, fn _b, x, y ->
          [Value.add(x, y, s32_typespec())]
        end)
      end
    end
  end

  describe "serialization" do
    test "run" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, s32_typespec())

      exec =
        compile([s32_typespec(), s32_typespec()], [], [s32_typespec()], fn _, x, y ->
          [Value.add(x, y, s32_typespec())]
        end)

      dumped = Executable.dump(exec)
      exec = Executable.load(client(), dumped)

      assert [[a = %DeviceBuffer{}]] = EXLA.Executable.run(exec, [[t1, t2]], [])
      assert <<2::32-native>> == DeviceBuffer.read(a)
    end
  end

  defp s32_typespec(), do: Typespec.tensor({:s, 32}, {})
end

defmodule EXLA.ExecutableFeedTest do
  # infeed/outfeed are global resources, so they either
  # need to be locked or we cannot run them concurrently.
  use ExUnit.Case, async: false

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Client
  alias EXLA.Typespec
  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value
  import EXLAHelpers

  describe "infeed/outfeed" do
    test "successfully sends to/from device asynchronously" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Typespec.tensor({:s, 32}, {}))

      assert res =
               Task.async(fn ->
                 run_one([], [], [t.typespec], fn b ->
                   token = Value.create_token(b)

                   {new_token, [val]} = Value.infeed(token, [t.typespec])

                   outfeed_val = Value.add(val, val, s32_typespec())
                   _outfeed_token = Value.outfeed(outfeed_val, new_token)
                   [Value.add(outfeed_val, val, s32_typespec())]
                 end)
               end)

      assert :ok = Client.to_infeed(client(), 0, [{t.data, t.typespec}])
      assert from_outfeed(client(), 0, Typespec.tensor({:s, 32}, {})) == <<2::32-native>>

      assert [a = %DeviceBuffer{}] = Task.await(res)
      assert DeviceBuffer.read(a) == <<3::32-native>>
    end

    test "successfully sends to/from device asynchronously in a loop" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Typespec.tensor({:s, 32}, {}))

      token_shape = Typespec.token()

      assert res =
               Task.async(fn ->
                 run_one([], [], [t.typespec], fn b ->
                   token = Value.create_token(b)

                   arg_shapes = [token_shape, t.typespec]

                   {condition_region, [_token, val]} = Function.push_region(b, arg_shapes)
                   zero = Value.constant(b, [0], s32_typespec())
                   Value.return(b, [Value.not_equal(val, zero, Typespec.tensor({:u, 8}, {}))])
                   Function.pop_region(b)

                   {body_region, [body_token, val]} = Function.push_region(b, arg_shapes)

                   body_token = Value.outfeed(Value.add(val, val, s32_typespec()), body_token)
                   {body_token, [input]} = Value.infeed(body_token, [t.typespec])

                   Value.return(b, [body_token, input])
                   Function.pop_region(b)

                   {token, [val]} = Value.infeed(token, [t.typespec])
                   [_token, result] = Value.while(b, condition_region, body_region, [token, val])

                   [result]
                 end)
               end)

      assert :ok = Client.to_infeed(client(), 0, [{<<1::32-native>>, t.typespec}])
      assert from_outfeed(client(), 0, Typespec.tensor({:s, 32}, {})) == <<2::32-native>>

      assert :ok = Client.to_infeed(client(), 0, [{<<2::32-native>>, t.typespec}])
      assert from_outfeed(client(), 0, Typespec.tensor({:s, 32}, {})) == <<4::32-native>>

      assert :ok = Client.to_infeed(client(), 0, [{<<0::32-native>>, t.typespec}])

      assert [a = %DeviceBuffer{}] = Task.await(res)
      assert DeviceBuffer.read(a) == <<0::32-native>>
    end
  end

  defp s32_typespec(), do: Typespec.tensor({:s, 32}, {})

  defp from_outfeed(client, device_id, typespec) do
    ref = make_ref()
    Client.from_outfeed(client, device_id, [typespec], self(), ref)

    receive do
      {^ref, msg} -> msg
    end
  end
end
