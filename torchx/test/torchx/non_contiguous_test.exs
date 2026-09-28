defmodule Torchx.NonContiguousTest do
  @moduledoc """
  Operations on tensors whose LibTorch storage is not a contiguous row-major buffer.

  `Nx.transpose/2` is a permute view. `Nx.slice/4` copies into a new tensor, so
  the slice tests check that the copy follows logical order, including strides.
  """

  use Torchx.Case, async: true

  defp assert_op(tensor, fun) do
    expected = tensor |> Nx.backend_copy(Nx.BinaryBackend) |> then(fun)
    assert_equal(fun.(tensor), expected)
  end

  test "transpose then multiply" do
    tensor = Nx.iota({2, 3, 4}, type: :s32) |> Nx.transpose()

    assert_op(tensor, &Nx.multiply(&1, 2))
    assert_op(tensor, &Nx.add(&1, &1))
  end

  test "slice then reduce" do
    tensor = Nx.iota({2, 3, 4}, type: :s32)

    assert_op(tensor, fn t ->
      t |> Nx.slice([0, 1, 0], [2, 1, 4]) |> Nx.sum()
    end)

    assert_op(tensor, fn t ->
      t |> Nx.slice([0, 0, 0], [2, 3, 4], strides: [1, 2, 1]) |> Nx.sum()
    end)

    transposed = Nx.transpose(tensor)

    assert_op(transposed, fn t ->
      t |> Nx.slice([1, 0, 0], [2, 3, 1]) |> Nx.sum(axes: [0])
    end)
  end

  test "conversion of a transposed tensor" do
    tensor = Nx.iota({2, 3, 4}, type: :f32) |> Nx.transpose()
    binary = Nx.backend_copy(tensor, Nx.BinaryBackend)

    assert Nx.to_binary(tensor) == Nx.to_binary(binary)
    assert Nx.to_binary(tensor, limit: 5) == Nx.to_binary(binary, limit: 5)
  end

  test "unit dimensions" do
    tensor = Nx.iota({1, 4, 1}, type: :f32) |> Nx.transpose(axes: [2, 0, 1])

    assert_op(tensor, &Nx.multiply(&1, 3))
    assert_op(tensor, &Nx.reshape(&1, {4}))
  end

  test "transpose then reshape" do
    tensor = Nx.iota({2, 3, 4}, type: :s32) |> Nx.transpose()

    assert_op(tensor, &Nx.reshape(&1, {6, 4}))
    assert_op(tensor, &Nx.reshape(&1, {8, 3}))
  end

  test "element-wise ops on views" do
    left = Nx.iota({2, 3, 4}, type: :f32) |> Nx.transpose()
    right = Nx.iota({2, 3, 4}, type: :f32) |> Nx.reverse(axes: [1]) |> Nx.transpose()

    binary_left = Nx.backend_copy(left, Nx.BinaryBackend)
    binary_right = Nx.backend_copy(right, Nx.BinaryBackend)

    assert_equal(Nx.multiply(left, right), Nx.multiply(binary_left, binary_right))
    assert_equal(Nx.add(left, 1), Nx.add(binary_left, 1))

    rows = Nx.iota({4, 5}, type: :s32)

    assert_op(rows, fn t ->
      t |> Nx.slice([0, 1], [4, 3], strides: [2, 1]) |> Nx.add(1)
    end)
  end
end
