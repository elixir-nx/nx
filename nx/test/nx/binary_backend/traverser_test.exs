defmodule Nx.BinaryBackend.TraverserTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.Traverser

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

  test "Enum.count/1 works" do
    trav = Traverser.build({2, 2, 2, 2, 2, 2, 2}, [0])
    assert Traverser.size(trav) == 128
  end

  test "works for examples case" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = [0, 3, 6]
    trav = Traverser.build(shape, aggregate: axes)

    assert Traverser.to_flat_list(trav) == List.flatten(@example_expected)
  end

  test "works for {2, 3} and {3, 2} cases" do
    shape1 = {2, 3}
    axes1 = [1]
    t1 = Nx.iota(shape1, type: {:u, 8})

    exp1 = Nx.BinaryBackend.aggregate_axes(t1.data.state, axes1, t1.shape, 8)

    assert t1.data.state == <<0, 1, 2, 3, 4, 5>>

    trav1 = Traverser.build(shape1, aggregate: axes1)
    assert Traverser.to_flat_list(trav1) == exp1 |> Enum.join() |> to_charlist()

    shape2 = {3, 2}
    axes2 = [0]
    t2 = Nx.iota(shape2, type: {:u, 8})

    exp2 = Nx.BinaryBackend.aggregate_axes(t2.data.state, axes2, t2.shape, 8)
    trav2 = Traverser.build(shape2, aggregate: axes2)
    assert Traverser.to_flat_list(trav2) == exp2 |> Enum.join() |> to_charlist()
  end

  test "reduce_aggregates works" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = [0, 3, 6]
    trav = Traverser.build(shape, aggregate: axes)
    out = Traverser.reduce_aggregates(trav, [], fn agg, acc -> [agg | acc] end)
    assert Enum.reverse(out) == @example_expected
  end

  test "works with no :aggregate axes" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    trav = Traverser.build(shape, aggregate: [])
    assert Traverser.to_flat_list(trav) == Enum.to_list(0..127)
  end

  test "simple transpose works for {2, 3}" do
    trav = Traverser.build({2, 3}, transpose: [1, 0])
    cur = Nx.to_flat_list(Nx.transpose(Nx.iota({2, 3}, type: {:u, 8})))
    out = Traverser.to_flat_list(trav)
    assert out == cur
    assert out == [0, 3, 1, 4, 2, 5]
  end

  test "simple transpose works for {3, 2}" do
    trav = Traverser.build({3, 2}, transpose: [1, 0])
    cur = Nx.to_flat_list(Nx.transpose(Nx.iota({3, 2}, type: {:u, 8})))
    out = Traverser.to_flat_list(trav)
    assert out == cur
    assert out == [0, 2, 4, 1, 3, 5]
  end

  test "reverse works" do
    trav = Traverser.build({2, 2, 2}, reverse: true)
    assert Traverser.to_flat_list(trav) == Enum.map(7..0, fn i -> i end)
  end

  defp trav_dot(shape1, axes1, shape2, axes2) do
    trav1 = Traverser.build(shape1, aggregate: axes1)
    trav2 = Traverser.build(shape2, aggregate: axes2)

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
