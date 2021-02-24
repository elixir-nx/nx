defmodule TorchxTest do
  use ExUnit.Case, async: true

  doctest Torchx

  alias Torchx.Backend, as: TB

  describe "tensor" do
    test "add" do
      a = Nx.tensor([[1, 2], [3, 4]], backend: TB)
      b = Nx.tensor([[5, 6], [7, 8.0]], backend: TB)

      c = Nx.add(a, b)

      assert Nx.backend_transfer(c) == Nx.tensor([[6.0, 8.0], [10.0, 12.0]])
    end

    test "dot" do
      a = Nx.tensor([[1, 2], [3, 4]], backend: TB)
      b = Nx.tensor([[5, 6], [7, 8]], backend: TB)

      c = Nx.dot(a, b)

      assert Nx.backend_transfer(c) == Nx.tensor([[19, 22], [43, 50]])
    end
  end
end
