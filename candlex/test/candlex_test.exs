defmodule CandlexTest do
  use ExUnit.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      check(255, :u8)
      check(100_002, :u32)
      check(-101, :s64)
      check(1.11, :f32)
      check([1, 2, 3], :f32)
      check(-0.002, :f64)
      check([1, 2], :u32)
      check([[1, 2], [3, 4]], :u32)
      check([[1, 2, 3, 4], [5, 6, 7, 8]], :u32)
      check([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], :u32)
      check([0, 255], :u8)
      check([-0.5, 0.88], :f32)
      check([-0.5, 0.88], :f64)
    end
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
