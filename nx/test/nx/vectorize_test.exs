defmodule Nx.VectorizeTest do
  use ExUnit.Case, async: true

  @base Nx.tensor([
          [[0, 1, 2]],
          [[3, 4, 5]],
          [[6, 7, 8]]
        ])

  @vectorized Nx.vectorize(@base, :rows)

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

  describe "binary operations" do
    test "left addition by scalar" do
      result = Nx.add(2, @vectorized)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "right addition by scalar" do
      result = Nx.add(@vectorized, 2)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "left addition by rank-1" do
      result = Nx.add(Nx.tensor([2]), @vectorized)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "right addition by rank-1" do
      result = Nx.add(@vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "left addition by rank-2" do
      result = Nx.add(Nx.tensor([[1], [2]]), @vectorized)
      assert result.shape == {2, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([
                   [[1, 2, 3], [2, 3, 4]],
                   [[4, 5, 6], [5, 6, 7]],
                   [[7, 8, 9], [8, 9, 10]]
                 ]),
                 :rows
               )
    end

    test "right addition by rank-2" do
      result = Nx.add(@vectorized, Nx.tensor([[1], [2]]))
      assert result.shape == {2, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([
                   [[1, 2, 3], [2, 3, 4]],
                   [[4, 5, 6], [5, 6, 7]],
                   [[7, 8, 9], [8, 9, 10]]
                 ]),
                 :rows
               )
    end

    test "addition by vectorized with same axes" do
      assert Nx.vectorize(Nx.add(@base, @base), :rows) == Nx.add(@vectorized, @vectorized)
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

      assert result == Nx.add(@vectorized, v2)
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

      assert result == Nx.add(@vectorized, v2)
    end
  end
end
