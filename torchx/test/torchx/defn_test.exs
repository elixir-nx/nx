defmodule Torchx.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  alias Nx.Tensor, as: T
  alias Torchx.Backend, as: TB

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

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
end
