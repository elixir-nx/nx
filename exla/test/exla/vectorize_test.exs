defmodule Nx.VectorizeTest do
  use EXLA.Case, async: true

  import Nx.Defn

  setup do
    backend = Nx.default_backend(EXLA.Backend)

    on_exit(fn ->
      Nx.default_backend(backend)
    end)

    base =
      Nx.tensor(
        [
          [[0, 1, 2]],
          [[3, 4, 5]],
          [[6, 7, 8]]
        ],
        backend: EXLA.Backend
      )

    vectorized = Nx.vectorize(base, :rows)

    base_math =
      Nx.tensor(
        [
          [[0.1, 0.2, 0.3]],
          [[0.4, 0.5, 0.6]],
          [[0.7, 0.8, 0.9]]
        ],
        backend: EXLA.Backend
      )

    vectorized_math = Nx.vectorize(base_math, :rows)

    %{base: base, vectorized: vectorized, base_math: base_math, vectorized_math: vectorized_math}
  end

  defn add_n(x, y), do: Nx.add(x, y)

  def add(x, y) do
    EXLA.jit_apply(&add_n/2, [x, y])
  end

  defn equal_n(x, y), do: Nx.equal(x, y)

  def equal(x, y) do
    EXLA.jit_apply(&equal_n/2, [x, y])
  end

  describe "unary math ops" do
    for {name, _} <- Nx.Shared.unary_math_funs(), name != :acosh do
      defn_name = :"#{name}_n"
      defn unquote(defn_name)(x), do: Nx.unquote(name)(x)

      def unquote(name)(x) do
        EXLA.jit_apply(&unquote(defn_name)(&1), [x])
      end

      test "Nx.#{name}/1 works on vectorized tensor", %{
        base_math: base_math,
        vectorized_math: vectorized_math
      } do
        result =
          base_math
          |> Nx.unquote(name)()
          |> Nx.vectorize(:rows)

        assert_equal(result, unquote(name)(vectorized_math))
      end
    end

    test "Nx.acosh/1 works on vectorized tensor", %{
      base_math: base_math,
      vectorized_math: vectorized_math
    } do
      fun = EXLA.jit(&(&1 |> Nx.add(2) |> Nx.acosh()))

      expected = fun.(base_math) |> Nx.vectorize(:rows)
      result = fun.(vectorized_math)

      assert_equal(expected, result)
    end
  end

  describe "binary operations" do
    test "left addition by scalar", %{vectorized: vectorized} do
      result = add(2, vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "right addition by scalar", %{vectorized: vectorized} do
      result = add(vectorized, 2)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "left addition by rank-1", %{vectorized: vectorized} do
      result = add(Nx.tensor([2]), vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "right addition by rank-1", %{vectorized: vectorized} do
      result = add(vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "left addition by rank-2", %{vectorized: vectorized} do
      result = add(Nx.tensor([[1], [2]]), vectorized)
      assert result.shape == {2, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([
            [[1, 2, 3], [2, 3, 4]],
            [[4, 5, 6], [5, 6, 7]],
            [[7, 8, 9], [8, 9, 10]]
          ]),
          :rows
        )
      )
    end

    test "right addition by rank-2", %{vectorized: vectorized} do
      result = add(vectorized, Nx.tensor([[1], [2]]))
      assert result.shape == {2, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([
            [[1, 2, 3], [2, 3, 4]],
            [[4, 5, 6], [5, 6, 7]],
            [[7, 8, 9], [8, 9, 10]]
          ]),
          :rows
        )
      )
    end

    test "addition by vectorized with same axes", %{base: base, vectorized: vectorized} do
      assert_equal(
        Nx.vectorize(Nx.add(base, base), :rows),
        add(vectorized, vectorized)
      )
    end

    test "addition by vectorized with different axes", %{vectorized: vectorized, base: base} do
      v2 = Nx.vectorize(base, :cols)

      result =
        Nx.stack([
          Nx.add(base, Nx.tensor([[0, 1, 2]])),
          Nx.add(base, Nx.tensor([[3, 4, 5]])),
          Nx.add(base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert_equal(result, add(vectorized, v2))
    end

    test "addition by vectorized with common axes", %{vectorized: vectorized, base: base} do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([base, base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.add(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert_equal(result, add(vectorized, v2))
    end
  end

  describe "predicate operations" do
    test "left Nx.equal by scalar", %{vectorized: vectorized} do
      result = equal(2, vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "right Nx.equal by scalar", %{vectorized: vectorized} do
      result = equal(vectorized, 2)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "left Nx.equal by rank-1", %{vectorized: vectorized} do
      result = equal(Nx.tensor([2]), vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "right Nx.equal by rank-1", %{vectorized: vectorized} do
      result = equal(vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "left Nx.equal by rank-2", %{vectorized: vectorized} do
      result = equal(Nx.tensor([[1], [2]]), vectorized)
      assert result.shape == {2, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor(
            [
              [[0, 1, 0], [0, 0, 1]],
              [[0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0]]
            ],
            type: :u8
          ),
          :rows
        )
      )
    end

    test "right Nx.equal by rank-2", %{vectorized: vectorized} do
      result = equal(vectorized, Nx.tensor([[1], [2]]))
      assert result.shape == {2, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor(
            [
              [[0, 1, 0], [0, 0, 1]],
              [[0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0]]
            ],
            type: :u8
          ),
          :rows
        )
      )
    end

    test "Nx.equal by vectorized with same axes", %{vectorized: vectorized, base: base} do
      assert_equal(
        Nx.vectorize(Nx.broadcast(Nx.tensor(1, type: :u8), base), :rows),
        equal(vectorized, vectorized)
      )
    end

    test "Nx.equal by vectorized with different axes", %{vectorized: vectorized, base: base} do
      v2 = Nx.vectorize(base, :cols)

      result =
        Nx.stack([
          Nx.equal(base, Nx.tensor([[0, 1, 2]])),
          Nx.equal(base, Nx.tensor([[3, 4, 5]])),
          Nx.equal(base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert_equal(result, equal(vectorized, v2))
    end

    test "Nx.equal by vectorized with common axes", %{vectorized: vectorized, base: base} do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([base, base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.equal(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert_equal(result, equal(vectorized, v2))
    end
  end
end
