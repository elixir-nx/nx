defmodule Nx.VectorizeTest do
  use EXLA.Case, async: true

  import Nx.Defn

  @base Nx.tensor([
          [[0, 1, 2]],
          [[3, 4, 5]],
          [[6, 7, 8]]
        ])

  @vectorized Nx.vectorize(@base, :rows)

  @base_math Nx.tensor([
               [[0.1, 0.2, 0.3]],
               [[0.4, 0.5, 0.6]],
               [[0.7, 0.8, 0.9]]
             ])

  @vectorized_math Nx.vectorize(@base_math, :rows)

  setup do
    backend = Nx.default_backend(EXLA.Backend)

    on_exit(fn ->
      Nx.default_backend(backend)
    end)
  end

  defn add_n(x, y), do: Nx.add(x, y)

  def add(x, y) do
    EXLA.jit_apply(&add_n/2, [x, y])
  end

  defn equal_n(x, y), do: Nx.equal(x, y)

  def equal(x, y) do
    EXLA.jit_apply(&equal_n/2, [x, y])
  end

  describe "vectorize" do
    test "adds new vectorization axes to the end of the list" do
      v = Nx.vectorize(@vectorized, :cols)
      assert v.vectorized_axes == [rows: 3, cols: 1]
      assert v.shape == {3}
    end

    # TO-DO: re-write value inspect to support vectorization
    @tag :skip
    test "inspect works as expected"
  end

  describe "unary math ops" do
    for {name, _} <- Nx.Shared.unary_math_funs(), name != :acosh do
      defn_name = :"#{name}_n"
      defn unquote(defn_name)(x), do: Nx.unquote(name)(x)

      def unquote(name)(x) do
        EXLA.jit_apply(&unquote(defn_name)(&1), [x])
      end

      test "Nx.#{name}/1 works on vectorized tensor" do
        result =
          @base_math
          |> Nx.unquote(name)()
          |> Nx.vectorize(:rows)

        assert_equal(result, unquote(name)(@vectorized_math))
      end
    end

    test "Nx.acosh/1 works on vectorized tensor" do
      expected =
        @base_math
        |> Nx.add(1)
        |> Nx.acosh()
        |> Nx.vectorize(:rows)

      result = EXLA.jit_apply(&(&1 |> Nx.add(1) |> Nx.acosh()), [@vectorized_math])

      assert_equal(expected, result)
    end
  end

  describe "binary operations" do
    test "left addition by scalar" do
      result = add(2, @vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "right addition by scalar" do
      result = add(@vectorized, 2)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "left addition by rank-1" do
      result = add(Nx.tensor([2]), @vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "right addition by rank-1" do
      result = add(@vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "left addition by rank-2" do
      result = add(Nx.tensor([[1], [2]]), @vectorized)
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

    test "right addition by rank-2" do
      result = add(@vectorized, Nx.tensor([[1], [2]]))
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

    test "addition by vectorized with same axes" do
      assert Nx.vectorize(Nx.add(@base, @base), :rows) == add(@vectorized, @vectorized)
    end

    test "addition by vectorized with different axes" do
      v2 = Nx.vectorize(@base, :cols)

      result =
        Nx.stack([
          Nx.add(@base, Nx.tensor([[0, 1, 2]])),
          Nx.add(@base, Nx.tensor([[3, 4, 5]])),
          Nx.add(@base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert_equal(result, add(@vectorized, v2))
    end

    test "addition by vectorized with common axes" do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([@base, @base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.add(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert_equal(result, add(@vectorized, v2))
    end
  end

  describe "predicate operations" do
    test "left Nx.equal by scalar" do
      result = equal(2, @vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "right Nx.equal by scalar" do
      result = equal(@vectorized, 2)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "left Nx.equal by rank-1" do
      result = equal(Nx.tensor([2]), @vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "right Nx.equal by rank-1" do
      result = equal(@vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
          :rows
        )
      )
    end

    test "left Nx.equal by rank-2" do
      result = equal(Nx.tensor([[1], [2]]), @vectorized)
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

    test "right Nx.equal by rank-2" do
      result = equal(@vectorized, Nx.tensor([[1], [2]]))
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

    test "Nx.equal by vectorized with same axes" do
      assert_equal(
        Nx.vectorize(Nx.broadcast(Nx.tensor(1, type: :u8), @base), :rows),
        equal(@vectorized, @vectorized)
      )
    end

    test "Nx.equal by vectorized with different axes" do
      v2 = Nx.vectorize(@base, :cols)

      result =
        Nx.stack([
          Nx.equal(@base, Nx.tensor([[0, 1, 2]])),
          Nx.equal(@base, Nx.tensor([[3, 4, 5]])),
          Nx.equal(@base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert_equal(result, equal(@vectorized, v2))
    end

    test "Nx.equal by vectorized with common axes" do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([@base, @base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.equal(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert_equal(result, equal(@vectorized, v2))
    end
  end
end
