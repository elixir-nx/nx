defmodule Nx.BinaryBackend.WeightedShapeTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.WeightedShape

  doctest WeightedShape

  describe "build/4" do
    test "default args" do
      expected = [{2, 60}, {3, 20}, {4, 5}, {5, 1}]
      assert WeightedShape.build({2, 3, 4, 5}) == expected
      assert WeightedShape.build({2, 3, 4, 5}, 1, :none, 1) == expected
    end
  end

  describe "dilate/2" do
    test "matches build with an int" do
      built = WeightedShape.build({2, 3, 4, 5}, 1, :none, 3)
      ws = WeightedShape.build({2, 3, 4, 5})
      dilated = WeightedShape.dilate(ws, 3)
      expected = [{2, 180}, {3, 60}, {4, 15}, {5, 3}]
      assert built == expected
      assert dilated == expected
    end

    test "matches build with a list" do
      dilations = [3, 2, 1, 4]
      built = WeightedShape.build({2, 3, 4, 5}, 1, :none, dilations)
      ws = WeightedShape.build({2, 3, 4, 5})
      dilated = WeightedShape.dilate(ws, dilations)
      expected = [{2, 180}, {3, 40}, {4, 5}, {5, 4}]
      assert built == expected
      assert dilated == expected
    end

    test "does not dilate dimensions of length 1" do
      built = WeightedShape.build({5, 1, 5}, 1, :none, 10)
      ws = WeightedShape.build({5, 1, 5})
      dilated = WeightedShape.dilate(ws, 10)
      expected = [{5, 50}, {1, 5}, {5, 10}]
      assert built == expected
      assert dilated == expected
    end
  end

  describe "limit/2" do
    test "matches build" do
      built = WeightedShape.build({2, 3, 4, 5}, 1, {2, 2, 2, 2})
      ws = WeightedShape.build({2, 3, 4, 5})
      assert ws == [{2, 60}, {3, 20}, {4, 5}, {5, 1}]
      limited = WeightedShape.limit(ws, {2, 2, 2, 2})
      expected = [{2, 60}, {2, 20}, {2, 5}, {2, 1}]
      assert built == expected
      assert limited == expected
    end
  end
end
