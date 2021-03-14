defmodule Torchx.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  # TODO: Add a test using rewrite_types where
  # the result of an operation (say Nx.add/2)
  # is lowered to mismatch PyTorch type and
  # therefore explicit casting will be required.
  # We don't want to add this test right now
  # because we want to raise on type mismatches
  # as we develop the lib.

  describe "scalar" do
    defn float_scalar(x), do: Nx.add(1.0, x)
    defn integer_scalar(x), do: Nx.add(1, x)
    defn broadcast_scalar(x), do: Nx.add(Nx.broadcast(1, {2, 2}), x)

    test "works" do
      assert float_scalar(0.0) |> Nx.backend_transfer() ==
               Nx.tensor(1.0, backend: Nx.BinaryBackend)

      assert integer_scalar(0) |> Nx.backend_transfer() ==
               Nx.tensor(1, backend: Nx.BinaryBackend)

      assert broadcast_scalar(0) |> Nx.backend_transfer() ==
               Nx.broadcast(Nx.tensor(1, backend: Nx.BinaryBackend), {2, 2})
    end
  end
end
