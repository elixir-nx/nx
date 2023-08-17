defmodule CandlexTest do
  use ExUnit.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      Nx.tensor(100_002, type: :u32, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      Nx.tensor([1, 2], type: :u32, backend: Candlex.Backend)
      |> IO.inspect()
      |> Nx.to_binary()
      |> IO.inspect()

      # Nx.tensor([[1, 2], [3, 4]], type: :u32, backend: Candlex.Backend)
      # |> IO.inspect()
      # |> Nx.to_binary()
      # |> IO.inspect()

      # assert Nx.backend_transfer(tensor) == Nx.tensor(1, type: :u32, backend: Nx.BinaryBackend)
    end
  end
end
