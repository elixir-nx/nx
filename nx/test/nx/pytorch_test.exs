defmodule Nx.PytorchTest do
  use ExUnit.Case, async: true

  alias Nx.PytorchBackend

  describe "tensor" do
    test "add" do
      a = Nx.tensor([[1, 2], [3, 4]], backend: PytorchBackend)
      b = Nx.tensor([[5, 6], [7, 8.0]], backend: PytorchBackend)

      c = Nx.add(a, b)

      assert Nx.to_flat_list(c) == [6.0, 8.0, 10.0, 12.0]
    end
  end
end
