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

  describe "infeed custom call dtypes" do


    for {type, data} <- [
          {{:u, 8}, <<42>>},
          {{:s, 8}, <<-5::signed-8>>},
          {{:s, 16}, <<-123::signed-16-native>>},
          {{:u, 16}, <<123::unsigned-16-native>>},
          {{:u, 32}, <<123_456::unsigned-32-little>>},
          {{:u, 64}, <<123_456_789::unsigned-64-little>>},
          {{:s, 64}, <<-123_456_789::signed-64-little>>},
          {{:c, 64}, <<1.5::float-little-32, -2.5::float-little-32>>},
          {{:c, 128}, <<1.5::float-little-64, -2.5::float-little-64>>}
        ] do
      test "custom infeed #{inspect(type)}" do
        first_data_typespec = Typespec.tensor(unquote(Macro.escape(type)), {})
        second_data_typespec = Typespec.tensor({:s, 32}, {2})

        pid =
          start_supervised!(
            {Agent, fn -> [unquote(data), <<42::32-native, 1337::32-native>>] end}
          )

        NifCall.run(EXLA.NifCall.Runner, &infeed_callback(&1, pid), fn nif_call_tag ->
          tag_bin = :erlang.term_to_binary(nif_call_tag)
          tag_spec = Typespec.tensor({:u, 8}, {byte_size(tag_bin)})
          tag_buf = BinaryBuffer.from_binary(tag_bin, tag_spec)

          exec =
            compile([tag_spec], [], [first_data_typespec, second_data_typespec], fn _b,
                                                                                    tag_mlir ->
              {_next_tag, [res1, res2]} = Value.infeed_custom(tag_mlir, [first_data_typespec, second_data_typespec])
              [res1, res2]
            end)

          assert [[res1 = %DeviceBuffer{}, res2 = %DeviceBuffer{}]] =
                   EXLA.Executable.run(exec, [[tag_buf]])

          assert DeviceBuffer.read(res1) == unquote(data)
          assert DeviceBuffer.read(res2) == <<42::32-native, 1337::32-native>>
        end)
      end
    end
  end

  describe "outfeed custom call" do
    test "outfeeds s32 to an encoded pid" do
      exec =
        compile([], [], [s32_typespec()], fn b ->
          val = Value.constant(b, [21], s32_typespec())
          out = Value.add(val, val, s32_typespec())
          Value.outfeed_custom([out], b)
          [out]
        end)

      assert [[res = %DeviceBuffer{}]] = EXLA.Executable.run(exec, [[]])
      assert DeviceBuffer.read(res) == <<42::32-native>>

      assert_receive bin when is_list(bin)
      assert length(bin) == 1
      assert hd(bin) == <<42::32-native>>
    end

    test "outfeeds f32 to an encoded pid" do
      t_spec = Typespec.tensor({:f, 32}, {})

      exec =
        compile([], [], [t_spec], fn b ->
          one_point_five = Value.constant(b, [1.5], t_spec)
          out = Value.add(one_point_five, one_point_five, t_spec)
          Value.outfeed_custom([out], b)
          [out]
        end)

      assert [[res = %DeviceBuffer{}]] = EXLA.Executable.run(exec, [[]])
      assert DeviceBuffer.read(res) == <<3.0::float-native-32>>

      assert_receive bin when is_list(bin)
      assert length(bin) == 1
      assert hd(bin) == <<3.0::float-native-32>>
    end

    for {type, data, val} <- [
          {{:u, 8}, <<42>>, 42},
          {{:s, 8}, <<-5::signed-8>>, -5},
          {{:s, 16}, <<-123::signed-16-native>>, -123},
          {{:u, 16}, <<123::unsigned-16-native>>, 123},
          {{:u, 32}, <<123_456::unsigned-32-native>>, 123_456},
          {{:u, 64}, <<123_456_789::unsigned-64-native>>, 123_456_789},
          {{:s, 64}, <<-123_456_789::signed-64-native>>, -123_456_789},
          {{:f, 16}, <<1.5::float-native-16>>, 1.5},
          {{:bf, 16}, Nx.bf16(1.5) |> Nx.to_binary(), 1.5},
          {{:f, 64}, <<3.5::float-native-64>>, 3.5},
          {{:c, 64}, <<1.5::float-native-32, -2.5::float-native-32>>,
           %Complex{re: 1.5, im: -2.5}},
          {{:c, 128}, <<1.5::float-native-64, -2.5::float-native-64>>,
           %Complex{re: 1.5, im: -2.5}}
        ] do
      test "outfeeds #{inspect(type)} to an encoded pid" do
        t_spec = Typespec.tensor(unquote(type), {})

        exec =
          compile([], [], [t_spec], fn b ->
            const = Value.constant(b, [unquote(Macro.escape(val))], t_spec)
            Value.outfeed_custom([const], b)
            [const]
          end)

        assert [[res = %DeviceBuffer{}]] = EXLA.Executable.run(exec, [[]])
        assert DeviceBuffer.read(res) == unquote(data)

        assert_receive bin when is_list(bin)
        assert length(bin) == 1
        assert hd(bin) == unquote(data)
      end
    end
  end

  describe "variadic infeed/outfeed custom calls" do
    test "variadic infeed with multiple tensor types" do
      data_list = [
        <<42>>,  # u8
        <<-123::signed-16-native>>,  # s16
        <<3.14::float-native-32>>  # f32
      ]

      typespecs = [
        Typespec.tensor({:u, 8}, {}),
        Typespec.tensor({:s, 16}, {}),
        Typespec.tensor({:f, 32}, {})
      ]

      pid = start_supervised!({Agent, fn -> data_list end})

      NifCall.run(EXLA.NifCall.Runner, &infeed_callback(&1, pid), fn nif_call_tag ->
        tag_bin = :erlang.term_to_binary(nif_call_tag)
        tag_spec = Typespec.tensor({:u, 8}, {byte_size(tag_bin)})
        tag_buf = BinaryBuffer.from_binary(tag_bin, tag_spec)

        exec =
          compile([tag_spec], [], typespecs, fn _b, tag_mlir ->
            {_next_tag, results} = Value.infeed_custom(tag_mlir, typespecs)
            results
          end)

        assert [results] = EXLA.Executable.run(exec, [[tag_buf]])
        assert length(results) == 3

        [res1, res2, res3] = Enum.map(results, &DeviceBuffer.read/1)
        assert res1 == <<42>>
        assert res2 == <<-123::signed-16-native>>
        assert res3 == <<3.14::float-native-32>>
      end)
    end

    test "variadic outfeed with multiple tensor types" do
      typespecs = [
        Typespec.tensor({:u, 8}, {}),
        Typespec.tensor({:s, 16}, {}),
        Typespec.tensor({:f, 32}, {})
      ]

      exec =
        compile([], [], typespecs, fn b ->
          val1 = Value.constant(b, [42], Enum.at(typespecs, 0))
          val2 = Value.constant(b, [-123], Enum.at(typespecs, 1))
          val3 = Value.constant(b, [3.14], Enum.at(typespecs, 2))

          Value.outfeed_custom([val1, val2, val3], b)
          [val1, val2, val3]
        end)

      assert [results] = EXLA.Executable.run(exec, [[]])
      assert length(results) == 3

      # Should receive a list of binaries
      assert_receive tensor_list when is_list(tensor_list)
      assert length(tensor_list) == 3

      [bin1, bin2, bin3] = tensor_list
      assert bin1 == <<42>>
      assert bin2 == <<-123::signed-16-native>>
      assert bin3 == <<3.14::float-native-32>>
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

  def infeed_callback(action, pid) do
    case action do
      :next_variadic ->
        # Variadic callback - return list of tensors
        Agent.get_and_update(pid, fn data_list ->
          tag = NifCall.Runner.register(EXLA.NifCall.Runner, &infeed_callback(&1, pid))
          tag_bin = :erlang.term_to_binary(tag)
          {{data_list, tag_bin}, []}
        end)

      # Backward compatibility - treat any other atom as variadic with single item
      _ ->
        Agent.get_and_update(pid, fn
          [h | tl] ->
            tag = NifCall.Runner.register(EXLA.NifCall.Runner, &infeed_callback(&1, pid))
            tag_bin = :erlang.term_to_binary(tag)
            {{[h], tag_bin}, tl}

          data_list when is_list(data_list) ->
            tag = NifCall.Runner.register(EXLA.NifCall.Runner, &infeed_callback(&1, pid))
            tag_bin = :erlang.term_to_binary(tag)
            {{data_list, tag_bin}, []}
        end)
    end
  end
end
