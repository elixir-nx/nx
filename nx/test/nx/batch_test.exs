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

  describe "merge" do
    test "returns new for empty list" do
      assert Nx.Batch.merge([]) == Nx.Batch.new()
    end

    test "merges padding" do
      batch = Nx.Batch.merge([Nx.Batch.new() |> Nx.Batch.pad(2), Nx.Batch.new() |> Nx.Batch.pad(3)])
      assert batch.pad == 5
    end

    test "merges empty batch" do
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      assert Nx.Batch.merge(Nx.Batch.new(), batch) == batch
      assert Nx.Batch.merge(batch, Nx.Batch.new()) == batch
    end
  end

  describe "errors" do
    test "raises when jitting empty batch" do
      assert_raise ArgumentError,
                   "cannot traverse/jit/compile Nx.Batch without entries",
                   fn -> Nx.Defn.jit_apply(&Function.identity/1, [Nx.Batch.new()]) end
    end

    test "raises on stack mismatch" do
      assert_raise ArgumentError,
                   ~r"cannot add to batch due to incompatible tensors/containers",
                   fn -> Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5])]) end
    end

    test "raises on concatenate mismatch" do
      assert_raise ArgumentError,
                   ~r"cannot add to batch due to incompatible tensors/containers",
                   fn -> Nx.Batch.concatenate([Nx.tensor([[1, 2, 3]]), Nx.tensor([[4, 5]])]) end
    end
  end
end
