defmodule EXLA.ExecutableTest do
  use ExUnit.Case, async: true

  alias EXLA.{BinaryBuffer, DeviceBuffer, Executable, Op, Shape, Builder}
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
    test "succeeds with no inputs and default options" do
      assert [%BinaryBuffer{data: <<1::32-native>>}] =
               run_one([], fn b ->
                 Op.tuple(b, [Op.constant_r0(b, 1, {:s, 32})])
               end)
    end

    test "succeeds with 2 inputs and default options" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [%BinaryBuffer{data: <<2::32-native>>}] =
               run_one([t1, t2], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
    end

    test "get_dimension_size/2" do
      tests = [
        # {input enumerable, {input shape}, dimension, expected size}
        {1..10, {10}, 0, 10},
        {1..10, {2, 5}, 0, 2},
        {1..12, {4, 3}, 1, 3},
        {1..12, {3, 2, 2}, 2, 2}
      ]

      Enum.each(tests, fn {data, shape, dim, expected} ->
        want = make_buffer(expected, {:s, 32}, {})
        operand = make_buffer(data, {:s, 32}, shape)

        [have] =
          run_one([operand], fn b, operand ->
            Op.tuple(b, [Op.get_dimension_size(operand, dim)])
          end)

        assert have.data == want.data
      end)
    end

    defp build_reduce_sum_computation(name) do
      b = Builder.new(name)
      sum_shape = Shape.make_shape({:s, 32}, {})

      lhs = Op.parameter(b, 0, sum_shape, "a")
      rhs = Op.parameter(b, 1, sum_shape, "b")
      sum_ast = Op.add(lhs, rhs)
      Builder.build(sum_ast)
    end

    test "set_dimension_size/2" do
      # Output is asserted using the same technique as in
      # https://www.tensorflow.org/xla/operation_semantics#setdimensionsize

      [
        # {input enumerable, {input shape}, {dimension, new size}, {expected
        # reduce data, expected reduce shape}}
        {1..10, {10}, {0, 5}, {15, {}}},
        {1..10, {10}, {0, 2}, {3, {}}},

        # 3=1+2, 13=6+7 !
        # 1 2 | 3 4 5
        # 6 7 | 8 9 10
        {1..10, {2, 5}, {1, 2}, {[3, 13], {2}}}
      ]
      |> Enum.with_index()
      |> Enum.each(fn {{data, shape, {dim, new_size}, {exp_data, exp_shape}}, index} ->
        want = make_buffer(exp_data, {:s, 32}, exp_shape)
        operand = make_buffer(data, {:s, 32}, shape)
        new_size = make_buffer(new_size, {:s, 32}, {})
        acc = make_buffer(0, {:s, 32}, {})
        sum = build_reduce_sum_computation("reduce-#{inspect(index)}")

        [have] =
          run_one([operand, new_size, acc], fn b, operand, new_size, acc ->
            resized = Op.set_dimension_size(operand, new_size, dim)

            # NOTE: the reduction is performed at the same dimension as the
            # dynamic dimension change is.
            Op.tuple(b, [Op.reduce(resized, acc, sum, {dim})])
          end)

        assert have.data == want.data
      end)
    end

    test "set_dimension_size/2 shrink and squeeze special case" do
      # Test designed to understand what happens if a dimension is first
      # shrinked then squeezed. Two outcomes are possible: either the data cut
      # in the shrink phase is truncated and lost or the undelying data is left
      # untouched. The latter case is what is happening here.

      operand = make_buffer(1..10, {:s, 32}, {10})
      new_size_shrink = make_buffer(2, {:s, 32}, {})
      new_size_squeeze = make_buffer(4, {:s, 32}, {})
      want_shrinked = make_buffer(3, {:s, 32}, {})
      want_squeezed = make_buffer(10, {:s, 32}, {})

      acc_shrink = make_buffer(0, {:s, 32}, {})
      acc_squeeze = make_buffer(0, {:s, 32}, {})
      sum = build_reduce_sum_computation("reduce")

      params = [
        operand,
        new_size_shrink,
        new_size_squeeze,
        acc_shrink,
        acc_squeeze
      ]

      [have_shrinked, have_squeezed] =
        run_one(params, fn b,
                           operand,
                           new_size_shrink,
                           new_size_squeeze,
                           acc_shrink,
                           acc_squeeze ->
          shrinked = Op.set_dimension_size(operand, new_size_shrink, 0)
          squeezed = Op.set_dimension_size(shrinked, new_size_squeeze, 0)

          Op.tuple(b, [
            Op.reduce(shrinked, acc_shrink, sum, {0}),
            Op.reduce(squeezed, acc_squeeze, sum, {0})
          ])
        end)

      assert have_shrinked.data == want_shrinked.data
      assert have_squeezed.data == want_squeezed.data
    end

    test "set_dimension_size/2 leaves tensor shape untouched" do
      # Test designed to ensure that set_dimension_size operations do not
      # actually change the tensor's shape.
      operand = make_buffer(1..10, {:s, 32}, {10})
      new_size = make_buffer(6, {:s, 32}, {})

      [have] =
        run_one([operand, new_size], fn b, operand, new_size ->
          Op.tuple(b, [Op.set_dimension_size(operand, new_size, 0)])
        end)

      assert have.shape.dims == {10}
    end

    test "dynamic_reshape/4" do
      [
        # {input enumerable, {input shape}, [reshape upper bounds], [reshape
        # dynamic dimensions], {expected reduce data, expected reduce shape}}
        {1..10, {10}, [10], [6], {21, {}}},
        {1..10, {10}, [5, 2], [5, 2], {[3, 7, 11, 15, 19], {5}}},
        # Look carefully: underlying data is flattened, truncated and finally
        # re-shaped! One could also expect the output to be {3, 9, 15, 21}.
        {1..12, {12}, [4, 3], [4, 2], {[3, 7, 11, 15], {4}}}
      ]
      |> Enum.with_index()
      |> Enum.each(fn {{data, shape, bounds, new_sizes, {exp_data, exp_shape}}, index} ->
        want = make_buffer(exp_data, {:s, 32}, exp_shape)

        operand = make_buffer(data, {:s, 32}, shape)
        new_sizes = Enum.map(new_sizes, &make_buffer(&1, {:s, 32}, {}))
        is_dynamic = List.duplicate(true, length(bounds))

        acc = make_buffer(0, {:s, 32}, {})
        sum = build_reduce_sum_computation("reduce-#{inspect(index)}")

        # Compilation phase. Helper functions are not used as the input
        # bounds/sizes are lists of variables sizes and run_one/compile are not
        # designed for that.

        builder = EXLA.Builder.new("test-#{inspect(index)}")
        shapes = Enum.map([operand, acc | new_sizes], fn x -> x.shape end)

        # Avoid operand, acc and new_sizes variable shadowing.
        op =
          (fn ->
             [operand, acc | new_sizes] =
               shapes
               |> Enum.with_index()
               |> Enum.map(fn {shape, pos} ->
                 EXLA.Op.parameter(builder, pos, shape, <<?a + pos>>)
               end)

             reshaped = Op.dynamic_reshape(operand, new_sizes, bounds, is_dynamic)
             reduced = Op.reduce(reshaped, acc, sum, {length(bounds) - 1})
             Op.tuple(builder, [reduced])
           end).()

        [[result]] =
          op
          |> EXLA.Builder.build()
          |> EXLA.Computation.compile(client(), shapes)
          |> EXLA.Executable.run([[operand, acc | new_sizes]])

        assert result.data == want.data
      end)
    end

    test "succeeds when data is preloaded" do
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
               run_one([t1, t2], [keep_on_device: true], fn b, x, y ->
                 Op.tuple(b, [Op.add(x, y)])
               end)

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [keep_on_device: true], fn b, x, y ->
                 Op.tuple(b, [Op.add(x, y)])
               end)

      assert DeviceBuffer.read(t1) == <<1::32-native>>
      assert DeviceBuffer.read(t2) == <<1::32-native>>
    end

    test "succeeds with keep_on_device is true" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [%DeviceBuffer{}] =
               run_one([t1, t2], [keep_on_device: true], fn b, x, y ->
                 Op.tuple(b, [Op.add(x, y)])
               end)
    end

    test "succeeds with data from a previous run" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      exec = compile([t1.shape, t2.shape], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
      assert [[t3 = %DeviceBuffer{}]] = Executable.run(exec, [[t1, t2]], keep_on_device: true)
      assert [[%BinaryBuffer{data: <<4::32-native>>}]] = Executable.run(exec, [[t3, t3]])
    end

    test "succeeds with mixed data" do
      t1 =
        DeviceBuffer.place_on_device(
          <<1::32-native>>,
          Shape.make_shape({:s, 32}, {}),
          client(),
          0
        )

      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [%BinaryBuffer{data: <<3::32-native>>}] =
               run_one([t1, t2], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
    end

    test "succeeds with tuple return" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [%BinaryBuffer{data: <<1::32-native>>}, %BinaryBuffer{data: <<2::32-native>>}] =
               run_one([t1, t2], fn b, x, y -> Op.tuple(b, [x, y]) end)
    end

    test "succeeds with tuple return and keep_on_device true" do
      t1 = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))
      t2 = BinaryBuffer.from_binary(<<2::32-native>>, Shape.make_shape({:s, 32}, {}))

      assert [a = %DeviceBuffer{}, b = %DeviceBuffer{}, c = %DeviceBuffer{}] =
               run_one([t1, t2], [keep_on_device: true], fn b, x, y ->
                 Op.tuple(b, [x, y, Op.add(x, y)])
               end)

      assert <<1::32-native>> == DeviceBuffer.read(a)
      assert <<2::32-native>> == DeviceBuffer.read(b)
      assert <<3::32-native>> == DeviceBuffer.read(c)
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
                 run_one([], [keep_on_device: true], fn b ->
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
                 run_one([], [keep_on_device: true], fn b ->
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
