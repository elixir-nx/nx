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
  end
end
