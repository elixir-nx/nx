defmodule CandlexTest do
  use ExUnit.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      tensor = Nx.tensor(100_002, type: :u32, backend: Candlex.Backend)

      tensor
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      Nx.tensor([1, 2], type: :u32, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      Nx.tensor([[1, 2], [3, 4]], type: :u32, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], type: :u32, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], type: :u32, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      assert Nx.backend_copy(tensor) == Nx.tensor(100_002, type: :u32, backend: Nx.BinaryBackend)
      assert Nx.backend_transfer(tensor) == Nx.tensor(100_002, type: :u32, backend: Nx.BinaryBackend)

      Nx.tensor([-0.5, 0.88], type: :f64, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()
    end
  end
end
