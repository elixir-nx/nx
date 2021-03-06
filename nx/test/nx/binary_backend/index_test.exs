defmodule Nx.BinaryBackend.IndexTest do
  use ExUnit.Case, async: true
  alias Nx.BinaryBackend.Index
  alias Nx.BinaryBackend.Weights

  doctest Index

  describe "project_on_axis/3" do
    test "works" do
      shape = {2, 3}
      weights = Weights.build(shape)
      weight1 = Weights.weight_of_axis(weights, 1)
      assert weight1 == 1
      weight0 = Weights.weight_of_axis(weights, 0)
      assert weight0 == 3

      assert Index.project_on_axis(weights, 1, 0) == 0
      assert Index.project_on_axis(weights, 1, 1) == 1
      assert Index.project_on_axis(weights, 1, 2) == 2
      assert Index.project_on_axis(weights, 1, 3) == 3
      assert Index.project_on_axis(weights, 1, 4) == 4
      assert Index.project_on_axis(weights, 1, 5) == 5

      assert Index.project_on_axis(weights, 0, 0) == 0
      assert Index.project_on_axis(weights, 0, 1) == 3
      assert Index.project_on_axis(weights, 0, 2) == 0
      assert Index.project_on_axis(weights, 0, 3) == 3
      assert Index.project_on_axis(weights, 0, 4) == 0
      assert Index.project_on_axis(weights, 0, 5) == 3
    end
  end
end