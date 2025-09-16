defmodule Torchx.DefnTest do
  use Torchx.Case, async: true

  import Nx.Defn
  alias Nx.Tensor, as: T
  alias Torchx.Backend, as: TB

  describe "creation" do
    defn iota, do: Nx.iota({10, 10})

    test "iota" do
      %T{data: %TB{}} = tensor = iota()
      assert Nx.backend_transfer(tensor) == Nx.iota({10, 10}, backend: Nx.BinaryBackend)
    end
  end

  describe "scalar" do
    defn float_scalar(x), do: Nx.add(1.0, x)
    defn integer_scalar(x), do: Nx.add(1, x)
    defn broadcast_scalar(x), do: Nx.add(Nx.broadcast(1, {2, 2}), x)

    test "float" do
      %T{data: %TB{}} = tensor = float_scalar(0.0)
      assert Nx.backend_transfer(tensor) == Nx.tensor(1.0, backend: Nx.BinaryBackend)
    end

    test "integer" do
      %T{data: %TB{}} = tensor = integer_scalar(0)
      assert Nx.backend_transfer(tensor) == Nx.tensor(1, backend: Nx.BinaryBackend)
    end

    test "broadcast" do
      %T{data: %TB{}} = tensor = broadcast_scalar(0)

      assert Nx.backend_transfer(tensor) ==
               Nx.broadcast(Nx.tensor(1, backend: Nx.BinaryBackend), {2, 2})
    end
  end

  describe "while" do
    defn factorial_tuple(x) do
      factorial = Nx.tensor(1, type: Nx.type(x))

      {factorial, _} =
        while {factorial, x}, Nx.greater(x, 1) do
          {factorial * x, x - 1}
        end

      factorial
    end

    test "factorial tuple" do
      assert factorial_tuple(5) |> Nx.backend_transfer() ==
               Nx.tensor(120, backend: Nx.BinaryBackend)

      assert factorial_tuple(10.0) |> Nx.backend_transfer() ==
               Nx.tensor(3_628_800.0, backend: Nx.BinaryBackend)
    end
  end

  describe "determinant" do
    defn det(t), do: Nx.LinAlg.determinant(t)

    test "works" do
      t = Nx.tensor([[2, 0], [0, 1]])
      assert_all_close(Nx.tensor(2.0), det(t))
    end
  end
end
