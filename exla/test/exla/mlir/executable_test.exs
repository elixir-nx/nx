defmodule EXLA.MLIR.ExecutableTest do
  use EXLA.Case, async: true

  import Nx.Defn

  setup do
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

  @broadcast_types [s: 8, u: 8, s: 64, u: 64, f: 32, f: 16, f: 64]
  @types [
    s: 8,
    s: 16,
    s: 32,
    s: 64,
    u: 8,
    u: 16,
    u: 32,
    u: 64,
    f: 16,
    f: 32,
    f: 64,
    bf: 16,
    c: 64,
    c: 128
  ]

  describe "create_function" do
    test "creates with tuple arguments" do
      result = EXLA.jit(fn {{t1}, {t2}} -> Nx.add(t1, t2) end, compiler_mode: :mlir).({{1}, {2}})
      assert_equal(result, Nx.tensor(3))
    end

    test "creates with tuple return" do
      result =
        EXLA.jit(fn t -> {Nx.as_type(t, :f32), Nx.as_type(t, :f16)} end, compiler_mode: :mlir).(1)

      assert_equal(result, {Nx.f32(1), Nx.f16(1)})
    end

    test "creates with mixed container result and input" do
      result =
        EXLA.jit(
          fn %{a: a, b: %{c: c, d: {d}}} ->
            %{a: {Nx.add(a, c), %{b: Nx.multiply(c, d)}}}
          end,
          compiler_mode: :mlir
        ).(%{a: 1, b: %{c: 10, d: {20}}})

      assert_equal(result, %{a: {11, %{b: 200}}})
    end
  end

  describe "bitcast" do
    test "converts between same bitwidth" do
      t = Nx.s32([1337, 42])
      result = Nx.Defn.jit_apply(&Nx.bitcast(&1, :f32), [t], compiler_mode: :mlir)

      assert_equal(
        result,
        Nx.from_binary(<<1337::32-integer-native, 42::32-integer-native>>, :f32)
      )
    end
  end

  describe "pad" do
    test "pads in all dims" do
      result =
        EXLA.jit(&Nx.pad(&1, &2, [{2, 3, 1}, {0, 0, 0}]), compiler_mode: :mlir).(
          Nx.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
          Nx.tensor(100)
        )

      assert_equal(
        result,
        Nx.tensor([
          [100, 100, 100],
          [100, 100, 100],
          [1, 1, 1],
          [100, 100, 100],
          [2, 2, 2],
          [100, 100, 100],
          [3, 3, 3],
          [100, 100, 100],
          [100, 100, 100],
          [100, 100, 100]
        ])
      )

      result =
        EXLA.jit(&Nx.pad(&1, &2, [{0, 0, 0}, {2, 3, 1}]), compiler_mode: :mlir).(
          Nx.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
          Nx.tensor(100)
        )

      assert_equal(
        result,
        Nx.tensor([
          [100, 100, 1, 100, 1, 100, 1, 100, 100, 100],
          [100, 100, 2, 100, 2, 100, 2, 100, 100, 100],
          [100, 100, 3, 100, 3, 100, 3, 100, 100, 100]
        ])
      )
    end
  end

  describe "convert" do
    for origin_type <- @types, dest_type <- @types do
      test "converts #{inspect(origin_type)} to #{inspect(dest_type)}" do
        t = Nx.tensor([[1], [2]], type: unquote(origin_type))
        expected = Nx.as_type(t, unquote(dest_type))

        result = Nx.Defn.jit_apply(&Nx.as_type(&1, unquote(dest_type)), [t])
        assert result.type == expected.type
        assert_equal(result, expected)
      end
    end

    for {k, _} = type <- @types, k in [:u, :s] do
      test "#{inspect(type)} max_finite to u64" do
        t = Nx.Constants.max_finite(unquote(type))

        expected = Nx.as_type(t, :u64)
        result = Nx.Defn.jit_apply(&Nx.as_type(&1, :u64), [t])

        assert result.type == expected.type
        assert_equal(result, expected)
      end
    end

    for {k, _} = type <- @types, k in [:u, :s] do
      test "#{inspect(type)} min_finite to f64" do
        t = Nx.Constants.min_finite(unquote(type))

        expected = Nx.as_type(t, :f64)
        result = Nx.Defn.jit_apply(&Nx.as_type(&1, :f64), [t])

        assert result.type == expected.type
        assert_equal(result, expected)
      end
    end
  end

  describe "binary ops" do
    @bin_ops [:add, :subtract, :multiply, :pow, :min] ++
               [:max, :remainder, :atan2, :equal, :not_equal] ++
               [:less, :less_equal, :greater, :greater_equal]
    for op <- @bin_ops do
      test "#{op}" do
        for type1 <- @broadcast_types,
            type2 <- @broadcast_types,
            explicit_broadcast <- [true, false] do
          function = fn t1, t2 -> Nx.unquote(op)(t1, t2) end

          t1 = Nx.iota({2, 3, 1}, type: type1)

          t2 =
            if explicit_broadcast do
              Nx.broadcast(Nx.tensor(2, type: type2), {2, 3, 1})
            else
              Nx.tensor(2, type: type2)
            end

          result_nx = Nx.Defn.jit_apply(function, [t1, t2], compiler: Nx.Defn.Evaluator)
          result_mlir = Nx.Defn.jit_apply(function, [t1, t2])

          if Nx.Type.float?(result_mlir.type) do
            assert result_nx.shape == result_mlir.shape
            assert result_nx.type == result_mlir.type
            # TO-DO (mlir): remove backend transfer when all_close is fully supported
            assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir),
              rtol: 1.0e-4
            )
          else
            assert_equal(result_nx, result_mlir)
          end
        end
      end
    end

    for op <- [:bitwise_and, :bitwise_or, :bitwise_xor] do
      test "#{op}" do
        for type1 <- @broadcast_types,
            type2 <- @broadcast_types,
            explicit_broadcast <- [true, false],
            not Nx.Type.float?(type1) and not Nx.Type.float?(type2) do
          function = fn t1, t2 -> Nx.unquote(op)(t1, t2) end

          t1 = Nx.iota({2, 3, 1}, type: type1)

          t2 =
            if explicit_broadcast do
              Nx.broadcast(Nx.tensor(2, type: type2), {2, 3, 1})
            else
              Nx.tensor(2, type: type2)
            end

          result_nx = Nx.Defn.jit_apply(function, [t1, t2], compiler: Nx.Defn.Evaluator)
          result_mlir = Nx.Defn.jit_apply(function, [t1, t2])

          assert_equal(result_nx, result_mlir)
        end
      end
    end

    for op <- [:left_shift, :right_shift], type <- [u: 8, s: 8] do
      test "#{op} #{inspect(type)}" do
        for explicit_broadcast <- [true, false] do
          function = fn t1, t2 -> Nx.unquote(op)(t1, t2) end

          t1 = Nx.iota({2, 3, 1}, type: unquote(type))

          t2 =
            if explicit_broadcast do
              Nx.broadcast(Nx.tensor(2, type: unquote(type)), {2, 3, 1})
            else
              Nx.tensor(2, type: unquote(type))
            end

          result_nx = Nx.Defn.jit_apply(function, [t1, t2], compiler: Nx.Defn.Evaluator)
          result_mlir = Nx.Defn.jit_apply(function, [t1, t2])

          assert_equal(result_nx, result_mlir)
        end
      end
    end
  end

  describe "unary ops" do
    @unary_ops [:abs, :exp, :expm1, :floor, :ceil, :round] ++
                 [:log, :log1p, :sign, :cosh, :sinh] ++
                 [:sqrt, :cbrt, :sin, :cos, :atan] ++
                 [:tanh, :sigmoid, :erf, :erfc, :rsqrt] ++
                 [:negate, :conjugate, :real, :imag]

    for op <- @unary_ops do
      test "#{op}" do
        function = fn t -> Nx.unquote(op)(t) end

        t =
          Nx.Defn.jit_apply(&Nx.add/2, [Nx.iota({2, 3, 1}, type: :f32), 1],
            compiler: Nx.Defn.Evaluator
          )

        result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t])

        assert result_nx.shape == result_mlir.shape
        assert result_nx.type == result_mlir.type
        # TO-DO (mlir): remove backend transfer when all_close is fully supported
        assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
      end
    end

    test "is_infinity" do
      function = fn t -> Nx.is_infinity(t) end

      t = Nx.tensor([:nan, 0, :infinity, :neg_infinity, :nan, 10])

      result_nx = Nx.u8([0, 0, 1, 1, 0, 0])
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      # TO-DO (mlir): remove backend transfer when all_close is fully supported
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    test "is_nan" do
      function = fn t -> Nx.is_nan(t) end

      t = Nx.tensor([:nan, 0, :infinity, :neg_infinity, :nan, 10])

      result_nx = Nx.u8([1, 0, 0, 0, 1, 0])
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      # TO-DO (mlir): remove backend transfer when all_close is fully supported
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    test "erf_inf" do
      function = fn t -> Nx.erf_inv(t) end

      t = Nx.erf(Nx.tensor([1, 1, 2, 3, 5, 8]))

      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      # TO-DO (mlir): remove backend transfer when all_close is fully supported
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    for op <- [:count_leading_zeros, :bitwise_not, :population_count] do
      test "#{op}" do
        function = fn t -> Nx.unquote(op)(t) end

        t = Nx.iota({2, 3, 1}, type: :s64)

        result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t])

        assert_equal(result_nx, result_mlir)
      end
    end

    test "acosh" do
      function = fn t -> Nx.acosh(t) end

      t =
        Nx.Defn.jit_apply(
          &Nx.divide(&3, Nx.add(&1, &2)),
          [Nx.iota({2, 3, 1}, type: :f32), 1, 100],
          compiler: Nx.Defn.Evaluator
        )

      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      # TO-DO (mlir): remove backend transfer when all_close is fully supported
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    for op <- [:acos, :atanh, :asin, :asinh] do
      test "#{op}" do
        function = fn t -> Nx.unquote(op)(t) end

        t =
          Nx.Defn.jit_apply(
            &Nx.divide(Nx.add(&1, &2), &3),
            [Nx.iota({2, 3, 1}, type: :f32), 1, 100],
            compiler: Nx.Defn.Evaluator
          )

        result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t])

        assert result_nx.shape == result_mlir.shape
        assert result_nx.type == result_mlir.type
        # TO-DO (mlir): remove backend transfer when all_close is fully supported
        assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
      end
    end

    # TO-DO (mlir): this case depends on broadcasting being available
    test "sign with unsigned input" do
      function = fn t -> Nx.sign(t) end

      t = Nx.tensor([0, 1, 2], type: :s8)

      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "constants" do
    test "iota" do
      for axis <- [0, 1, 2, nil] do
        function = fn -> Nx.iota({2, 3, 4}, axis: axis) end

        expected_result = Nx.Defn.jit_apply(function, [], compiler: Nx.Defn.Evaluator)
        mlir_result = Nx.Defn.jit_apply(function, [])

        assert_equal(expected_result, mlir_result)
      end
    end

    test "constant_r0" do
      for type <- @types do
        function = fn -> Nx.tensor(10, type: type) end

        expected_result = Nx.Defn.jit_apply(function, [], compiler: Nx.Defn.Evaluator)
        mlir_result = Nx.Defn.jit_apply(function, [])

        assert_equal(expected_result, mlir_result)
      end
    end

    test "constant_from_binary" do
      for type <- @types do
        function = fn -> Nx.tensor([[10], [20], [30]], type: type) end

        expected_result = Nx.Defn.jit_apply(function, [], compiler: Nx.Defn.Evaluator)
        mlir_result = Nx.Defn.jit_apply(function, [])

        assert_equal(expected_result, mlir_result)
      end
    end
  end

  describe "reshape" do
    test "works" do
      t = Nx.iota({24})
      function = fn t -> Nx.reshape(t, {2, 3, 4}) end
      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "slice" do
    test "works" do
      t = Nx.iota({2, 3, 4})

      function = fn t -> Nx.slice(t, [0, 0, 0], [2, 3, 1]) end
      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)

      function = fn t -> Nx.slice(t, [0, Nx.tensor(0), Nx.tensor(1)], [2, 3, 4]) end
      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "reverse" do
    test "works" do
      t = Nx.iota({2, 3, 4})
      axes = [[], 0, 1, 2]

      for ax1 <- axes, ax2 <- axes, ax3 <- axes do
        axes = Enum.uniq(List.flatten([ax1, ax2, ax3]))
        function = fn t -> Nx.reverse(t, axes: axes) end
        result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t])

        assert result_nx.shape == result_mlir.shape
        assert result_nx.type == result_mlir.type
        assert_equal(result_nx, result_mlir)
      end
    end
  end

  describe "tranpose" do
    test "works" do
      t = Nx.iota({2, 3, 4})
      axes = [0, 1, 2]

      for ax1 <- axes, ax2 <- axes, ax3 <- axes, ax1 != ax2 and ax1 != ax3 and ax2 != ax3 do
        axes = [ax1, ax2, ax3]
        function = fn t -> Nx.transpose(t, axes: axes) end
        result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t])

        assert result_nx.shape == result_mlir.shape
        assert result_nx.type == result_mlir.type
        assert_equal(result_nx, result_mlir)
      end
      |> then(fn x -> assert length(x) == 6 end)
    end
  end

  describe "dot_general" do
    test "works" do
      lhs = Nx.iota({2, 3, 4}, type: {:f, 32})
      rhs = Nx.iota({2, 3, 4}, type: {:f, 32})

      function = fn lhs, rhs -> Nx.dot(lhs, [2], [0], rhs, [2], [0]) end
      result_nx = Nx.Defn.jit_apply(function, [lhs, rhs], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [lhs, rhs])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "broadcast_in_dim" do
    test "works" do
      t = Nx.iota({1, 2, 3})

      function = fn tensor -> Nx.broadcast(tensor, {3, 2, 3}) end
      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])
      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "concatenate" do
    test "works" do
      inputs = [Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]), Nx.tensor([7, 8, 9])]

      function = fn tensors -> Nx.concatenate(tensors, axis: 0) end
      result_nx = Nx.Defn.jit_apply(function, [inputs], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [inputs])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "clamp" do
    test "works" do
      value = Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
      min = Nx.tensor(1)
      max = Nx.tensor(3)

      function = fn value, min, max -> Nx.clip(value, min, max) end
      result_nx = Nx.Defn.jit_apply(function, [value, min, max], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [value, min, max])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "select" do
    test "works" do
      pred = Nx.tensor([0, 1, 0, 1])
      on_true = Nx.tensor([1, 2, 3, 4])
      on_false = Nx.tensor([5, 6, 7, 8])

      function = fn x, y, z -> Nx.select(x, y, z) end

      result_nx =
        Nx.Defn.jit_apply(function, [pred, on_true, on_false], compiler: Nx.Defn.Evaluator)

      result_mlir = Nx.Defn.jit_apply(function, [pred, on_true, on_false])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      assert_equal(result_nx, result_mlir)
    end
  end

  describe "sort" do
    test "sorts with axis and direction" do
      for type <- [s: 64, u: 64, f: 32] do
        t = Nx.tensor([[0, 2, 1, 10], [10, 10, 20, 0]], type: type)

        result = EXLA.jit(&Nx.sort(&1, direction: :asc, axis: 0), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor(
            [
              [0, 2, 1, 0],
              [10, 10, 20, 10]
            ],
            type: type
          )
        )

        result = EXLA.jit(&Nx.sort(&1, direction: :asc, axis: 1), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor(
            [
              [0, 1, 2, 10],
              [0, 10, 10, 20]
            ],
            type: type
          )
        )

        result = EXLA.jit(&Nx.sort(&1, direction: :desc, axis: 0), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor(
            [
              [10, 10, 20, 10],
              [0, 2, 1, 0]
            ],
            type: type
          )
        )

        result = EXLA.jit(&Nx.sort(&1, direction: :desc, axis: 1), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor(
            [
              [10, 2, 1, 0],
              [20, 10, 10, 0]
            ],
            type: type
          )
        )
      end
    end
  end

  describe "argsort" do
    test "sorts with axis and direction" do
      for type <- [s: 64, u: 64, f: 32] do
        t = Nx.tensor([[0, 2, 1, 10], [10, 11, 20, 0]], type: type)

        result = EXLA.jit(&Nx.argsort(&1, direction: :asc, axis: 0), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor([
            [0, 0, 0, 1],
            [1, 1, 1, 0]
          ])
        )

        result = EXLA.jit(&Nx.argsort(&1, direction: :asc, axis: 1), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor([
            [0, 2, 1, 3],
            [3, 0, 1, 2]
          ])
        )

        result = EXLA.jit(&Nx.argsort(&1, direction: :desc, axis: 0), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor([
            [1, 1, 1, 0],
            [0, 0, 0, 1]
          ])
        )

        result = EXLA.jit(&Nx.argsort(&1, direction: :desc, axis: 1), compiler_mode: :mlir).(t)

        assert_equal(
          result,
          Nx.tensor([
            [3, 1, 2, 0],
            [2, 1, 0, 3]
          ])
        )
      end
    end
  end

  describe "indexed ops" do
    test "indexed_add" do
      t = Nx.iota({1, 2, 3})
      indices = Nx.tensor([[0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 2], [0, 1, 2]])
      updates = Nx.tensor([1, 3, 1, -2, 5])

      result = EXLA.jit(&Nx.indexed_add/3, compiler_mode: :mlir).(t, indices, updates)

      assert_equal(
        result,
        Nx.tensor([
          [
            [2, 1, 0],
            [3, 7, 10]
          ]
        ])
      )
    end

    test "indexed_put" do
      t = Nx.iota({1, 2, 3})
      indices = Nx.tensor([[0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 2], [0, 1, 2]])
      updates = Nx.tensor([1, 3, 1, -2, 5])

      result = EXLA.jit(&Nx.indexed_put/3, compiler_mode: :mlir).(t, indices, updates)

      assert_equal(
        result,
        Nx.tensor([
          [
            [1, 1, -2],
            [3, 3, 5]
          ]
        ])
      )
    end

    test "axes option support" do
      t = Nx.iota({1, 2, 3})
      indices = Nx.tensor([[0, 0], [0, 2]])
      updates = Nx.tensor([[0, 30], [20, 50]])

      result = EXLA.jit(&Nx.indexed_put(&1, &2, &3, axes: [0, 2])).(t, indices, updates)

      assert_equal(
        result,
        Nx.tensor([
          [
            [0, 1, 20],
            [30, 4, 50]
          ]
        ])
      )

      result = EXLA.jit(&Nx.indexed_add(&1, &2, &3, axes: [0, 2])).(t, indices, updates)

      assert_equal(
        result,
        Nx.tensor([
          [
            [0, 1, 22],
            [33, 4, 55]
          ]
        ])
      )
    end
  end

  describe "window_scatter" do
    test "window_scatter_max" do
      t =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      opts = [strides: [2, 3], padding: :valid]

      result =
        EXLA.jit(&Nx.window_scatter_max(&1, Nx.tensor([[2, 6], [3, 1]]), 0, {2, 3}, opts),
          compiler_mode: :mlir
        ).(t)

      assert_equal(
        result,
        Nx.tensor([
          [0, 0, 0, 0, 6, 0],
          [0, 0, 2, 0, 0, 0],
          [0, 0, 3, 0, 0, 0],
          [0, 0, 0, 0, 0, 1]
        ])
      )
    end

    test "window_scatter_min" do
      t =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      opts = [strides: [2, 3], padding: :valid]

      result =
        EXLA.jit(&Nx.window_scatter_min(&1, Nx.tensor([[2, 6], [3, 1]]), 0, {2, 3}, opts),
          compiler_mode: :mlir
        ).(t)

      assert_equal(
        result,
        Nx.tensor([
          [0, 2, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 6],
          [0, 0, 0, 0, 0, 1],
          [3, 0, 0, 0, 0, 0]
        ])
      )
    end
  end

  describe "fft" do
    test "fft and ifft" do
      t = Nx.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])

      result = EXLA.jit(&(&1 |> Nx.fft() |> Nx.ifft()), compiler_mode: :mlir).(t)
      assert_equal(result, Nx.as_type(t, :c64))
    end

    test "fft2 and ifft2" do
      t = Nx.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])

      result = EXLA.jit(&(&1 |> Nx.fft2() |> Nx.ifft2()), compiler_mode: :mlir).(t)
      assert_equal(result, Nx.as_type(t, :c64))
    end
  end

  describe "conv" do
    test "simple convolution" do
      right = Nx.iota({4, 1, 1, 1})
      left = Nx.iota({1, 1, 3, 3})
      result = EXLA.jit(&Nx.conv(&1, &2, strides: [1, 1]), compiler_mode: :mlir).(left, right)

      assert_equal(
        result,
        Nx.tensor([
          [
            [
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]
            ],
            [
              [0.0, 1.0, 2.0],
              [3.0, 4.0, 5.0],
              [6.0, 7.0, 8.0]
            ],
            [
              [0.0, 2.0, 4.0],
              [6.0, 8.0, 10.0],
              [12.0, 14.0, 16.0]
            ],
            [
              [0.0, 3.0, 6.0],
              [9.0, 12.0, 15.0],
              [18.0, 21.0, 24.0]
            ]
          ]
        ])
      )
    end
  end

  describe "reduce" do
    test "sum defaults" do
      for type <- @types do
        tensor = Nx.tensor([1, 2, 3, 4], type: type)

        function = &Nx.sum/1

        result_nx = Nx.Defn.jit_apply(function, [tensor], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [tensor])

        assert_equal(result_nx, result_mlir)
      end
    end

    test "sum custom axes" do
      tensor = Nx.tensor([[[1, 2, 3.0], [4, 5, 6]]])

      function = &Nx.sum(&1, axes: [0, 2])

      result_nx = Nx.Defn.jit_apply(function, [tensor], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [tensor])

      assert_equal(result_nx, result_mlir)
    end

    test "sum keep axes" do
      tensor = Nx.tensor([[[1, 2, 3.0], [4, 5, 6]]])

      function = &Nx.sum(&1, axes: [0, 2], keep_axes: true)

      result_nx = Nx.Defn.jit_apply(function, [tensor], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [tensor])

      assert_equal(result_nx, result_mlir)
    end
  end

  describe "map" do
    test "works" do
      for type <- @types do
        tensor = Nx.tensor([1, 2, 3, 4], type: type)

        function = fn t -> Nx.map(t, &Nx.add(&1, 1)) end

        result_nx = Nx.Defn.jit_apply(function, [tensor], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [tensor])

        assert_equal(result_nx, result_mlir)
      end
    end
  end

  describe "triangular_solve" do
    test "supports options" do
      a = Nx.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
      b = Nx.tensor([[3, 1], [2, -1], [1, -1]])

      result =
        EXLA.jit(&Nx.LinAlg.triangular_solve(&1, &2, left_side: true), compiler_mode: :mlir).(
          a,
          b
        )

      assert_equal(
        result,
        Nx.tensor([
          [3.0, 1.0],
          [2.0, -1.0],
          [1.0, -1.0]
        ])
      )

      result =
        EXLA.jit(&Nx.LinAlg.triangular_solve(&1, &2, left_side: true, lower: false),
          compiler_mode: :mlir
        ).(a, b)

      assert_equal(
        result,
        Nx.tensor([
          [1.0, 2.0],
          [1.0, 0.0],
          [1.0, -1.0]
        ])
      )
    end
  end

  describe "put_slice" do
    test "purely static starts" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      u = Nx.tensor([[7, 8], [9, 10]])

      result = EXLA.jit(&Nx.put_slice(&1, [0, 1], &2)).(t, u)

      assert_equal(
        result,
        Nx.tensor([
          [1, 7, 8],
          [4, 9, 10]
        ])
      )
    end

    test "purely dynamic starts" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      u = Nx.tensor([[7, 8], [9, 10]])

      result = EXLA.jit(&Nx.put_slice(&1, [Nx.tensor(0), Nx.tensor(1)], &2)).(t, u)

      assert_equal(
        result,
        Nx.tensor([
          [1, 7, 8],
          [4, 9, 10]
        ])
      )
    end

    test "mixed starts" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      u = Nx.tensor([[7, 8], [9, 10]])
      result = EXLA.jit(&Nx.put_slice(&1, [Nx.tensor(1), 1], &2), compiler_mode: :mlir).(t, u)

      assert_equal(
        result,
        Nx.tensor([
          [1, 7, 8],
          [4, 9, 10]
        ])
      )
    end
  end

  describe "gather" do
    test "works" do
      t = Nx.tensor([[1, 2], [3, 4]])
      idx = Nx.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 1]]])
      result = EXLA.jit(fn t, idx -> Nx.gather(t, idx) end, compiler_mode: :mlir).(t, idx)

      assert_equal(
        result,
        Nx.tensor([
          [4, 1],
          [3, 2]
        ])
      )
    end
  end

  describe "cond" do
    defn cond_single_clause(t, x) do
      pred = t == 1

      cond do
        pred ->
          t + 10 + pred

        true ->
          x - 20
      end
    end

    defn cond_multi_clause(t, x) do
      cond do
        t == 1 ->
          t + x

        t == 2 ->
          -t

        true ->
          x - 20
      end
    end

    test "single-clause" do
      f = EXLA.jit(&cond_single_clause/2, compiler_mode: :mlir)
      assert_equal(f.(Nx.tensor(1), Nx.tensor(10)), 12)
      assert_equal(f.(Nx.tensor(0), Nx.tensor(10.0)), -10.0)
    end

    test "multi-clause" do
      f = EXLA.jit(&cond_multi_clause/2, compiler_mode: :mlir)
      assert_equal(f.(Nx.tensor(1.0), Nx.tensor(10)), 11.0)
      assert_equal(f.(Nx.tensor(2), Nx.tensor(10.0)), -2.0)
      assert_equal(f.(Nx.tensor(3), Nx.tensor(10)), -10)
    end
  end
end
