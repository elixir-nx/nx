defmodule TorchxTest do
  use ExUnit.Case, async: true

  doctest Torchx

  alias Torchx.Backend, as: TB

  # Torch Tensor creation shortcut
  defp tt(data), do: Nx.tensor(data, backend: Torchx.Backend)

  defp assert_equal(tt, data), do: assert(Nx.backend_transfer(tt) == Nx.tensor(data))

  describe "tensor" do
    test "add" do
      a = tt([[1, 2], [3, 4]])
      b = tt([[5, 6], [7, 8.0]])

      c = Nx.add(a, b)

      assert_equal(c, [[6.0, 8.0], [10.0, 12.0]])
    end

    test "dot" do
      a = tt([[1, 2], [3, 4]])
      b = tt([[5, 6], [7, 8]])

      c = Nx.dot(a, b)

      assert_equal(c, [[19, 22], [43, 50]])
    end
  end
end
