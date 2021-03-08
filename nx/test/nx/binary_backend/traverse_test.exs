defmodule Nx.BinaryBackend.TraverserTest do
  use ExUnit.Case, async: true
  
  describe "Traverser Enumerable" do
    test "can enumerate the correct sequence of indices" do
      shape = {2, 2, 2, 2, 2, 2, 2}
      axes = [0, 3, 6]
      {_, sizeof} = type = {:u, 8}
      t = Nx.iota(shape, type: type)
      agg_axes = Nx.BinaryBackend.aggregate_axes(t.data.state, axes, shape, sizeof)
      expected =
        agg_axes
        |> Enum.join()
        |> to_charlist()
      
      trav = Nx.BinaryBackend.Traverser.build(shape, axes)

      assert Enum.to_list(trav) == expected
    end
  end
end