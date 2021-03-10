defmodule Nx.BinaryBackend.TraverserTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.Traverser

  test "Enum.count/1 works" do
    trav = Traverser.build({2, 2, 2, 2, 2, 2, 2}, [0])
    assert Enum.count(trav) == 128
  end

  test "works for examples case" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = [0, 3, 6]
    trav = Traverser.build(shape, axes)

    assert Enum.to_list(trav) ==
             List.flatten([
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
             ])
  end

  test "works for {2, 3} and {3, 2} cases" do
    shape1 = {2, 3}
    axes1 = [1]
    t1 = Nx.iota(shape1, type: {:u, 8})

    exp1 = Nx.BinaryBackend.aggregate_axes(t1.data.state, axes1, t1.shape, 8)

    assert t1.data.state == <<0, 1, 2, 3, 4, 5>>

    trav1 = Traverser.build(shape1, axes1)
    assert Enum.to_list(trav1) == exp1 |> Enum.join() |> to_charlist()

    shape2 = {3, 2}
    axes2 = [0]
    t2 = Nx.iota(shape2, type: {:u, 8})

    exp2 = Nx.BinaryBackend.aggregate_axes(t2.data.state, axes2, t2.shape, 8)
    trav2 = Traverser.build(shape2, axes2)
    assert Enum.to_list(trav2) == exp2 |> Enum.join() |> to_charlist()
  end

  test "agg_iter works" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = [0, 3, 6]
    trav = Traverser.build(shape, axes)
    aggs = Traverser.agg_iter(trav)
    agg0 = Enum.at(aggs, 0)
    agg_size = Enum.count(agg0)

    expected = Enum.chunk_every(expected(shape, axes), agg_size)
    out = Enum.map(aggs, fn agg -> Enum.to_list(agg) end)
    assert out == expected
  end

  test "works for no axes" do
    shape = {2, 2, 2, 2, 2, 2, 2}
    axes = []
    trav = Traverser.build(shape, axes)
    assert Enum.to_list(trav) == Enum.to_list(0..127)
  end

  test "simple transpose works" do
    trav = Traverser.build({2, 2, 2}, [0], transpose: [2, 1, 0])
    assert Enum.to_list(trav) == [0, 1, 4, 5, 2, 3, 6, 7]
    aggs = Traverser.agg_iter(trav)

    out =
      Enum.map(aggs, fn agg ->
        Enum.to_list(agg)
      end)

    expected = [[0, 1], [4, 5], [2, 3], [6, 7]]
    assert out == expected
  end

  test "reverse works" do
    trav = Traverser.build({2, 2, 2}, [], reverse: true)
    aggs = Traverser.agg_iter(trav)

    out =
      Enum.map(aggs, fn agg ->
        Enum.to_list(agg)
      end)

    expected = Enum.map(7..0, fn i -> [i] end)
    assert out == expected
  end

  defp trav_dot(shape1, axes1, shape2, axes2) do
    trav1 = Traverser.build(shape1, axes1)
    trav2 = Traverser.build(shape2, axes2)

    aggs1 = Traverser.agg_iter(trav1)
    aggs2 = Traverser.agg_iter(trav2)

    out =
      for agg1 <- aggs1, agg2 <- aggs2 do
        # without a zip_reduce for now
        agg1
        |> Enum.zip(agg2)
        |> Enum.map(fn {i1, i2} -> i1 * i2 end)
        |> Enum.sum()
      end

    List.flatten(out)
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

    assert trav_dot(shape1, axes1, shape2, axes2) == cur_out
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

    assert trav_dot(shape1, axes1, shape2, axes2) == cur_out
  end
end
