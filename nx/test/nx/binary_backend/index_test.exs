defmodule Nx.BinaryBackend.IndexTest do
  use ExUnit.Case, async: true
  alias Nx.BinaryBackend.Index
  alias Nx.BinaryBackend.Weights

  doctest Index

  describe "contract_axis/3" do
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

    test "can be used to zip axes" do
      shape1 = {2, 3}
      shape2 = {3, 2}
      weights1 = Weights.build(shape1)
      weights2 = Weights.build(shape2)
      
      range = Index.range(3)

      w1_a0 = Weights.weight_of_axis(weights1, 0)
      w1_a1 = Weights.weight_of_axis(weights1, 1)

      assert w1_a0 == 3
      assert w1_a1 == 1

      w2_a0 = Weights.weight_of_axis(weights2, 0)
      w2_a1 = Weights.weight_of_axis(weights2, 1)

      assert w2_a0 == 2
      assert w2_a1 == 1
    
      # contracted axes
      indices1_axis1 = for i <- range do
        Index.project_on_axis(weights1, 1, i)
      end

      indices2_axis0 = for i <- range do
        Index.project_on_axis(weights2, 0, i)
      end

      # normal axes
      indices1_axis0 = for i <- range do
        i + w1_a0
      end

      offset_by_leftward_axes2 = w2_a0
      indices2_axis1 = for i <- range do
        i + offset_by_leftward_axes2
      end
      
    
      assert [indices1_axis1, indices1_axis0] == [[0, 1, 2], [3, 4, 5]]
      assert [indices2_axis0, indices2_axis1] == [[0, 2, 4], [1, 3, 5]]
    end

    
  end
end