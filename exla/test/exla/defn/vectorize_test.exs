defmodule EXLA.Defn.VectorizeTest do
  use EXLA.Case, async: true

  import Nx.Defn

  setup do
    Nx.default_backend(EXLA.Backend)

    base =
      Nx.tensor([
        [[0, 1, 2]],
        [[3, 4, 5]],
        [[6, 7, 8]]
      ])

    vectorized = Nx.vectorize(base, :rows)
    %{base: base, vectorized: vectorized}
  end

  defn add_n(x, y), do: Nx.add(x, y)

  def add(x, y) do
    EXLA.jit_apply(&add_n/2, [x, y])
  end

  describe "addition" do
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

  test "squeeze" do
    assert_equal(
      EXLA.jit(&Nx.squeeze/1).(Nx.iota({1, 1, 2}) |> Nx.vectorize(:x)),
      Nx.iota({1, 2}) |> Nx.vectorize(:x)
    )
  end
end
