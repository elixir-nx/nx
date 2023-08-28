defmodule CandlexTest do
  use Candlex.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      check(255, type: :u8)
      check(100_002, type: :u32)
      check(-101, type: :s64)
      check(1.16, type: :f16)
      check(1.32, type: :f32)
      check([1, 2, 3], type: :f32)
      check(-0.002, type: :f64)
      check([1, 2], type: :u32)
      check([[1, 2], [3, 4]], type: :u32)
      check([[1, 2, 3, 4], [5, 6, 7, 8]], type: :u32)
      check([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], type: :u32)
      check([0, 255], type: :u8)
      check([-0.5, 0.88], type: :f32)
      check([-0.5, 0.88], type: :f64)
      check(2.16, type: :bf16)
    end

    test "named dimensions" do
      check([[1, 2, 3], [4, 5, 6]], names: [:x, :y])

      t([[1, 2, 3], [4, 5, 6]], names: [:x, :y])
      |> assert_equal(t([[1, 2, 3], [4, 5, 6]]))
    end

    test "tensor tensor" do
      t(t([1, 2, 3]))
      |> assert_equal(t([1, 2, 3]))
    end

    test "tril" do
      t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      |> Nx.tril()
      |> assert_equal(t([[1, 0, 0], [4, 5, 0], [7, 8, 9]]))
    end

    test "triu" do
      t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      |> Nx.triu()
      |> assert_equal(t([[1, 2, 3], [0, 5, 6], [0, 0, 9]]))
    end

    test "addition" do
      t([1, 2, 3])
      |> Nx.add(t([10, 20, 30]))
      |> assert_equal(t([11, 22, 33]))
    end

    test "iota" do
      Nx.iota({})
      |> assert_equal(t(0))

      Nx.iota({}, type: :f32)
      |> assert_equal(t(0.0))

      Nx.iota({5})
      |> assert_equal(t([0, 1, 2, 3, 4]))

      # TODO: Support iota with float
      # Nx.iota({5}, type: :f32)
      # |> assert_equal(t([0.0, 1.0, 2.0, 3.0, 4.0]))

      Nx.iota({2, 3})
      |> assert_equal(t([[0, 1, 2], [3, 4, 5]]))
    end

    test "max" do
      Nx.max(1, 2)
      |> assert_equal(t(2))

      Nx.max(1, t([1.0, 2.0, 3.0], names: [:data]))
      |> assert_equal(t([1.0, 2.0, 3.0]))

      t([[1], [2]], type: :f32, names: [:x, nil])
      |> Nx.max(t([[10, 20]], type: :f32, names: [nil, :y]))
      |> assert_equal(t([[10.0, 20.0], [10.0, 20.0]]))
    end

    test "min" do
      Nx.min(1, 2)
      |> assert_equal(t(1))

      Nx.min(1, t([1.0, 2.0, 3.0], names: [:data]))
      |> assert_equal(t([1.0, 1.0, 1.0]))

      t([[1], [2]], type: :f32, names: [:x, nil])
      |> Nx.min(t([[10, 20]], type: :f32, names: [nil, :y]))
      |> assert_equal(t([[1.0, 1.0], [2.0, 2.0]]))
    end

    test "multiply" do
      t([1, 2])
      |> Nx.multiply(t([3, 4]))
      |> assert_equal(t([3, 8]))

      t([[1], [2]])
      |> Nx.multiply(t([3, 4]))
      |> assert_equal(t([[3, 4], [6, 8]]))

      t([1, 2])
      |> Nx.multiply(t([[3], [4]]))
      |> assert_equal(t([[3, 6], [4, 8]]))
    end

    test "access" do
      tensor = t([[1, 2], [3, 4]])

      assert_equal(tensor[0], t([1, 2]))
      assert_equal(tensor[1], t([3, 4]))
    end
  end

  defp t(values, opts \\ []) do
    opts =
      [backend: Candlex.Backend]
      |> Keyword.merge(opts)

    Nx.tensor(values, opts)
  end

  defp check(value, opts \\ []) do
    tensor = t(value, opts)

    tensor
    |> IO.inspect()
    |> Nx.to_binary()
    |> IO.inspect()

    opts =
      [backend: Nx.BinaryBackend]
      |> Keyword.merge(opts)

    assert Nx.backend_copy(tensor) == t(value, opts)
    assert Nx.backend_transfer(tensor) == t(value, opts)
  end
end
