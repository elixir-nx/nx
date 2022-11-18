defmodule Nx.BatchTest do
  use ExUnit.Case, async: true
  doctest Nx.Batch

  test "stack + concatenate" do
    batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
    assert batch.size == 2

    batch = Nx.Batch.concatenate(batch, [Nx.tensor([[11, 12, 13], [14, 15, 16]])])
    assert batch.size == 4

    assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) ==
             Nx.tensor([[1, 2, 3], [4, 5, 6], [11, 12, 13], [14, 15, 16]])
  end

  test "raises on batch mismatch" do
    assert_raise ArgumentError,
                 ~r"cannot add to batch due to incompatible tensors/containers",
                 fn -> Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5])]) end

    assert_raise ArgumentError,
                 ~r"cannot add to batch due to incompatible tensors/containers",
                 fn -> Nx.Batch.concatenate([Nx.tensor([[1, 2, 3]]), Nx.tensor([[4, 5]])]) end
  end
end
