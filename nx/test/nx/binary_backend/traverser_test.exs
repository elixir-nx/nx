defmodule Nx.BinaryBackend.TraverserTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.WeightedShape

  @example_expected [
    [0, 1, 8, 9, 64, 65, 72, 73],
    [2, 3, 10, 11, 66, 67, 74, 75],
    [4, 5, 12, 13, 68, 69, 76, 77],
    [6, 7, 14, 15, 70, 71, 78, 79],
    [16, 17, 24, 25, 80, 81, 88, 89],
    [18, 19, 26, 27, 82, 83, 90, 91],
    [20, 21, 28, 29, 84, 85, 92, 93],
    [22, 23, 30, 31, 86, 87, 94, 95],
    [32, 33, 40, 41, 96, 97, 104, 105],
    [34, 35, 42, 43, 98, 99, 106, 107],
    [36, 37, 44, 45, 100, 101, 108, 109],
    [38, 39, 46, 47, 102, 103, 110, 111],
    [48, 49, 56, 57, 112, 113, 120, 121],
    [50, 51, 58, 59, 114, 115, 122, 123],
    [52, 53, 60, 61, 116, 117, 124, 125],
    [54, 55, 62, 63, 118, 119, 126, 127]
  ]

  describe "size/1" do
    test "works" do
      shape = {2, 2, 2, 2, 2, 2, 2}
      ws = WeightedShape.build(shape)
      size = Nx.size(shape)
      trav = Traverser.build(size, 1, ws)
      assert Traverser.size(trav) == 128
    end
  end

  test "works for examples case" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = [0, 3, 6]

    ws =
      shape
      |> WeightedShape.build()
      |> WeightedShape.aggregate(axes)

    size = Nx.size(shape)
    trav = Traverser.build(size, 1, ws)
    assert Traverser.to_flat_list(trav) == List.flatten(@example_expected)
  end

  test "works for {2, 3} and {3, 2} cases" do
    shape1 = {2, 3}
    axes1 = [1]
    t1 = Nx.iota(shape1, type: {:u, 8})

    exp1 = Nx.BinaryBackend.aggregate_axes(t1.data.state, axes1, t1.shape, 8)

    assert t1.data.state == <<0, 1, 2, 3, 4, 5>>

    ws1 =
      shape1
      |> WeightedShape.build()
      |> WeightedShape.aggregate(axes1)

    size1 = Nx.size(shape1)

    trav1 = Traverser.build(size1, 1, ws1)

    assert Traverser.to_flat_list(trav1) == exp1 |> Enum.join() |> to_charlist()

    shape2 = {3, 2}
    axes2 = [0]
    t2 = Nx.iota(shape2, type: {:u, 8})

    exp2 = Nx.BinaryBackend.aggregate_axes(t2.data.state, axes2, t2.shape, 8)

     ws2 =
      t2.shape
      |> WeightedShape.build()
      |> WeightedShape.aggregate(axes2)

    size2 = Nx.size(t2.shape)
    trav2 = Traverser.build(size2, 1, ws2)
    assert Traverser.to_flat_list(trav2) == exp2 |> Enum.join() |> to_charlist()
  end

  test "reduce_aggregates works" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = [0, 3, 6]
    ws =
      shape
      |> WeightedShape.build()
      |> WeightedShape.aggregate(axes)

    size = Nx.size(shape)

    trav = Traverser.build(size, 1, ws)

    out = Traverser.reduce_aggregates(trav, [], fn agg, acc -> [agg | acc] end)

    assert Enum.reverse(out) == @example_expected
  end

  test "works for aggregate with empty axes" do
    shape = {2, 2, 2, 2, 2, 2, 2}

    ws =
      shape
      |> WeightedShape.build()
      |> WeightedShape.aggregate([])

    size = Nx.size(shape)
    trav = Traverser.build(size, 1, ws)
    assert Traverser.to_flat_list(trav) == Enum.to_list(0..127)
  end

  test "simple transpose works for {2, 3}" do
    shape = {2, 3}
    ws =
      shape
      |> WeightedShape.build()
      |> WeightedShape.transpose([1, 0])

    size = Nx.size(shape)
    trav = Traverser.build(size, 1, ws)

    cur = Nx.to_flat_list(Nx.transpose(Nx.iota(shape, type: {:u, 8})))
    out = Traverser.to_flat_list(trav)
    assert out == cur
    assert out == [0, 3, 1, 4, 2, 5]
  end

  test "simple transpose works for {3, 2}" do
    shape = {3, 2}
    ws =
      shape
      |> WeightedShape.build()
      |> WeightedShape.transpose([1, 0])

    size = Nx.size(shape)
    trav = Traverser.build(size, 1, ws)

    cur = Nx.to_flat_list(Nx.transpose(Nx.iota(shape, type: {:u, 8})))
    out = Traverser.to_flat_list(trav)
    assert out == cur
    assert out == [0, 2, 4, 1, 3, 5]
  end

  describe "reverse/2" do
    test "results in a reversed order when all dims are reversed" do
      shape = {2, 2, 2}
      ws =
        shape
        |> WeightedShape.build()
        |> WeightedShape.reverse([0, 1, 2])

      size = Nx.size(shape)
      trav = Traverser.build(size, 1, ws)
      assert Traverser.to_flat_list(trav) == Enum.map(7..0, fn i -> i end)
    end

    test "flips a matrix's rows for axis 0" do
      shape = {3, 3}
      ws =
        shape
        |> WeightedShape.build()
        |> WeightedShape.reverse([0])

      size = Nx.size(shape)
      trav = Traverser.build(size, 1, ws)
      assert Traverser.to_flat_list(trav) == List.flatten([
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
      ])
    end

    test "flips a matrix's columns for axis 1" do
      shape = {3, 3}
      ws =
        shape
        |> WeightedShape.build()
        |> WeightedShape.reverse([1])

      size = Nx.size(shape)
      trav = Traverser.build(size, 1, ws)
      assert Traverser.to_flat_list(trav) == List.flatten([
        [2, 1, 0],
        [5, 4, 3],
        [8, 7, 6]
      ])
    end
  end

  defp trav_dot(shape1, axes1, shape2, axes2) do
    ws1 = WeightedShape.build(shape1)
    ws1 = WeightedShape.aggregate(ws1, axes1)
    size1 = Nx.size(shape1)
    trav1 = Traverser.build(size1, 1, ws1)

    ws2 = WeightedShape.build(shape2)
    ws2 = WeightedShape.aggregate(ws2, axes2)
    size2 = Nx.size(shape2)
    trav2 = Traverser.build(size2, 1, ws2)

    Traverser.zip_reduce_aggregates(
      trav1,
      trav2,
      [],
      0,
      fn i1, i2, acc -> acc + i1 * i2 end,
      fn inner_acc, outer_acc -> [outer_acc, inner_acc] end
    )
    |> List.flatten()
  end

  test "simplest dot works just like the current implementation" do
    shape1 = {2, 3}
    shape2 = {3, 2}
    t1 = Nx.iota(shape1, type: {:u, 8})
    t2 = Nx.iota(shape2, type: {:u, 8})

    axes1 = [1]
    axes2 = [0]

    cur_out = Nx.dot(t1, axes1, t2, axes2)
    cur_out = Nx.to_flat_list(cur_out)
    dot = trav_dot(shape1, axes1, shape2, axes2)
    assert List.flatten(dot) == cur_out
  end

  test "works on a dot without any contraction axes" do
    shape1 = {2, 2, 2}
    shape2 = {2, 2, 2}
    t1 = Nx.iota(shape1, type: {:u, 8})
    t2 = Nx.iota(shape2, type: {:u, 8})

    axes1 = []
    axes2 = []

    cur_out = Nx.dot(t1, axes1, t2, axes2)
    cur_out = Nx.to_flat_list(cur_out)
    dot = trav_dot(shape1, axes1, shape2, axes2)
    assert List.flatten(dot) == cur_out
  end
end
