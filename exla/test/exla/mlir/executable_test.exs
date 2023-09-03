defmodule EXLA.MLIR.ExecutableTest do
  use EXLA.Case, async: true

  setup do
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

  test "mvp" do
    # TO-DO (mlir): this will probably be reorganized in the end
    # This test is being added as an MVP for MLIR compilation

    t1 = Nx.broadcast(0.0, {2, 3, 1})
    t2 = Nx.broadcast(1.0, {2, 3, 1})

    result =
      Nx.Defn.jit_apply(
        fn t1, t2 ->
          t1
          |> Nx.add(t2)
          |> then(&{Nx.add(t2, &1), Nx.subtract(t2, &1)})
          |> then(&elem(&1, 0))
        end,
        [t1, t2]
      )

    # expected = {Nx.add(t2, t2), t1}
    expected = Nx.add(t2, t2)
    assert_equal(result, expected)
  end

  describe "convert" do
    @types [s: 8, s: 16, s: 32, s: 64, u: 8, u: 16, u: 32, u: 64, f: 16, f: 32, f: 64, bf: 16]

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
    # TO-DO (mlir): test for type casting

    for op <- @bin_ops do
      test "#{op}" do
        function = fn t1, t2 -> Nx.unquote(op)(t1, t2) end

        t1 = Nx.iota({2, 3, 1}, type: :f32)
        t2 = Nx.broadcast(Nx.tensor(2, type: :f32), {2, 3, 1})

        result_nx = Nx.Defn.jit_apply(function, [t1, t2], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t1, t2])

        assert_equal(result_nx, result_mlir)
      end
    end

    for op <- [:bitwise_and, :bitwise_or, :bitwise_xor] do
      test "#{op}" do
        function = fn t1, t2 -> Nx.unquote(op)(t1, t2) end

        t1 = Nx.iota({2, 3, 1}, type: :s64)
        t2 = Nx.broadcast(Nx.tensor(2, type: :s64), {2, 3, 1})

        result_nx = Nx.Defn.jit_apply(function, [t1, t2], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t1, t2])

        assert_equal(result_nx, result_mlir)
      end
    end

    for op <- [:left_shift, :right_shift], type <- [u: 8, s: 8] do
      test "#{op} #{inspect(type)}" do
        function = fn t1, t2 -> Nx.unquote(op)(t1, t2) end

        t1 = Nx.iota({2, 3, 1}, type: unquote(type))
        t2 = Nx.broadcast(Nx.tensor(2, type: unquote(type)), {2, 3, 1})

        result_nx = Nx.Defn.jit_apply(function, [t1, t2], compiler: Nx.Defn.Evaluator)
        result_mlir = Nx.Defn.jit_apply(function, [t1, t2])

        assert_equal(result_nx, result_mlir)
      end
    end
  end

  describe "unary ops" do
    @unary_ops [:abs, :exp, :expm1, :floor, :ceil, :round] ++
                 [:log, :log1p, :sign, :cosh, :sinh] ++
                 [:sqrt, :cbrt, :sin, :cos, :atan] ++
                 [:tanh, :sigmoid, :erf, :erfc, :rsqrt] ++
                 [:negate]

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
        # TO-DO (mlir): remove backend transfer
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
      # TO-DO (mlir): remove backend transfer
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    test "is_nan" do
      function = fn t -> Nx.is_nan(t) end

      t = Nx.tensor([:nan, 0, :infinity, :neg_infinity, :nan, 10])

      result_nx = Nx.u8([1, 0, 0, 0, 1, 0])
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      # TO-DO (mlir): remove backend transfer
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    test "erf_inf" do
      function = fn t -> Nx.erf_inv(t) end

      t = Nx.erf(Nx.tensor([1, 1, 2, 3, 5, 8]))

      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert result_nx.shape == result_mlir.shape
      assert result_nx.type == result_mlir.type
      # TO-DO (mlir): remove backend transfer
      assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
    end

    test "bitwise_not" do
      function = fn t -> Nx.bitwise_not(t) end

      t = Nx.iota({2, 3, 1}, type: :s64)

      result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
      result_mlir = Nx.Defn.jit_apply(function, [t])

      assert_equal(result_nx, result_mlir)
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
      # TO-DO (mlir): remove backend transfer
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
        # TO-DO (mlir): remove backend transfer
        assert_all_close(Nx.backend_transfer(result_nx), Nx.backend_transfer(result_mlir))
      end
    end

    # TO-DO (mlir): this case depends on broadcasting being available
    # test "sign with unsigned input" do
    #   function = fn t -> Nx.sign(t) end

    #   t = Nx.tensor([0, 1, 2], type: :u8)

    #   result_nx = Nx.Defn.jit_apply(function, [t], compiler: Nx.Defn.Evaluator)
    #   result_mlir = Nx.Defn.jit_apply(function, [t])

    #   assert result_nx.shape == result_mlir.shape
    #   assert result_nx.type == result_mlir.type
    #   assert_equal(result_nx, result_mlir)
    # end
  end

  describe "constants" do
    test "iota" do
      for axis <- [0, 1, 2] do
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
end
