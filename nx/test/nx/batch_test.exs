defmodule Nx.BatchTest do
  use ExUnit.Case, async: true
  doctest Nx.Batch

  defp realize(batch) do
    Nx.Defn.jit_apply(&Function.identity/1, [batch])
  end

  test "stack + concatenate" do
    batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
    assert batch.size == 2

    batch = Nx.Batch.concatenate(batch, [Nx.tensor([[11, 12, 13], [14, 15, 16]])])
    assert batch.size == 4

    assert realize(batch) == Nx.tensor([[1, 2, 3], [4, 5, 6], [11, 12, 13], [14, 15, 16]])
  end

  describe "merge" do
    test "returns new for empty list" do
      assert Nx.Batch.merge([]) == Nx.Batch.new()
    end

    test "merges padding" do
      batch =
        Nx.Batch.merge([Nx.Batch.new() |> Nx.Batch.pad(2), Nx.Batch.new() |> Nx.Batch.pad(3)])

      assert batch.pad == 5
    end

    test "merges empty batch" do
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      assert Nx.Batch.merge(Nx.Batch.new(), batch) == batch
      assert Nx.Batch.merge(batch, Nx.Batch.new()) == batch
    end
  end

  describe "split" do
    test "when n >= size" do
      batch =
        [Nx.tensor([1, 2]), Nx.tensor([3, 4, 5])]
        |> Nx.Batch.concatenate()
        |> Nx.Batch.pad(3)

      {left, right} = Nx.Batch.split(batch, 5)
      assert left.template == right.template
      assert left.size == 5
      assert left.pad == 0
      assert realize(left) == Nx.tensor([1, 2, 3, 4, 5])
      assert right.size == 0
      assert right.pad == 3

      {left, right} = Nx.Batch.split(batch, 6)
      assert left.template == right.template
      assert left.size == 5
      assert left.pad == 1
      assert realize(left) == Nx.tensor([1, 2, 3, 4, 5, 0])
      assert right.size == 0
      assert right.pad == 2

      {left, right} = Nx.Batch.split(batch, 10)
      assert left.template == right.template
      assert left.size == 5
      assert left.pad == 3
      assert realize(left) == Nx.tensor([1, 2, 3, 4, 5, 0, 0, 0])
      assert right.size == 0
      assert right.pad == 0
    end

    test "when n < size" do
      batch =
        [Nx.tensor([1, 2]), Nx.tensor([3, 4, 5])]
        |> Nx.Batch.concatenate()
        |> Nx.Batch.pad(1)

      {left, right} = Nx.Batch.split(batch, 1)
      assert left.template == right.template
      assert left.size == 1
      assert left.pad == 0
      assert realize(left) == Nx.tensor([1])
      assert right.size == 4
      assert right.pad == 1
      assert realize(right) == Nx.tensor([2, 3, 4, 5, 0])

      {left, right} = Nx.Batch.split(batch, 2)
      assert left.template == right.template
      assert left.size == 2
      assert left.pad == 0
      assert realize(left) == Nx.tensor([1, 2])
      assert right.size == 3
      assert right.pad == 1
      assert realize(right) == Nx.tensor([3, 4, 5, 0])

      {left, right} = Nx.Batch.split(batch, 3)
      assert left.template == right.template
      assert left.size == 3
      assert left.pad == 0
      assert realize(left) == Nx.tensor([1, 2, 3])
      assert right.size == 2
      assert right.pad == 1
      assert realize(right) == Nx.tensor([4, 5, 0])

      {left, right} = Nx.Batch.split(batch, 4)
      assert left.template == right.template
      assert left.size == 4
      assert left.pad == 0
      assert realize(left) == Nx.tensor([1, 2, 3, 4])
      assert right.size == 1
      assert right.pad == 1
      assert realize(right) == Nx.tensor([5, 0])
    end

    test "composite" do
      batch =
        [
          {Nx.tensor([11, 12]), Nx.tensor([21, 22])},
          {Nx.tensor([13, 14, 15]), Nx.tensor([23, 24, 25])}
        ]
        |> Nx.Batch.concatenate()
        |> Nx.Batch.pad(1)

      {left, right} = Nx.Batch.split(batch, 3)
      assert {_, _} = left.template
      assert left.template == right.template
      assert left.size == 3
      assert left.pad == 0
      assert realize(left) == {Nx.tensor([11, 12, 13]), Nx.tensor([21, 22, 23])}
      assert right.size == 2
      assert right.pad == 1
      assert realize(right) == {Nx.tensor([14, 15, 0]), Nx.tensor([24, 25, 0])}
    end
  end

  describe "errors" do
    test "raises when jitting empty batch" do
      assert_raise ArgumentError,
                   "cannot traverse/jit/compile Nx.Batch without entries",
                   fn -> realize(Nx.Batch.new()) end
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
