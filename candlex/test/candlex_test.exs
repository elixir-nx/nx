defmodule CandlexTest do
  use Candlex.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      check(255, :u8)
      check(100_002, :u32)
      check(-101, :s64)
      check(1.16, :f16)
      check(1.32, :f32)
      check([1, 2, 3], :f32)
      check(-0.002, :f64)
      check([1, 2], :u32)
      check([[1, 2], [3, 4]], :u32)
      check([[1, 2, 3, 4], [5, 6, 7, 8]], :u32)
      check([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], :u32)
      check([0, 255], :u8)
      check([-0.5, 0.88], :f32)
      check([-0.5, 0.88], :f64)
      check(2.16, :bf16)
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

  defp t(values, backend \\ Candlex.Backend) do
    Nx.tensor(values, backend: backend)
  end

  defp check(value, type) do
    tensor = Nx.tensor(value, type: type, backend: Candlex.Backend)

    tensor
    |> IO.inspect()
    |> Nx.to_binary()
    |> IO.inspect()

    assert Nx.backend_copy(tensor) == Nx.tensor(value, type: type, backend: Nx.BinaryBackend)
    assert Nx.backend_transfer(tensor) == Nx.tensor(value, type: type, backend: Nx.BinaryBackend)
  end
end
