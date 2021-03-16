defmodule Nx.BinaryBackend.ViewTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.View

  describe "build/1" do
    test "works" do
      assert View.build({2, 3}) == %View{
        weighted_shape: [{2, 3}, {3, 1}]
      }
    end
  end

  describe "transpose/2" do
    test "flags changed as true" do
      view = View.build({2, 3})
      assert View.has_changes?(view) == false
      
      view_t = View.transpose(view, [0, 1])
      assert View.has_changes?(view_t) == true
    end

    test "sorted, consecutive axes do not change the weighted shape" do
      view = View.build({2, 3, 4})
      view_t = View.transpose(view, [0, 1, 2])
      assert view_t.weighted_shape == view.weighted_shape
    end

    test "unsorted axes swaps the position of weighted_shapes" do
      view = View.build({2, 3, 4})
      view_t = View.transpose(view, [2, 0, 1])
      assert view_t.weighted_shape == [{4, 1}, {2, 12}, {3, 4}]
    end
  end

  describe "reverse/2" do
    @tag :skip
    test "flips a matrix vertically for axes [1]"

    @tag :skip
    test "flips a matrix horizontally for axes [0]"

    @tag :skip
    test "reverses the order of a matrix for axes [0, 1]"
  end
end
