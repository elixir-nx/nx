defmodule Nx.BinaryBackend.ViewTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.View

  describe "build/1" do
    test "works" do
      assert View.build({2, 3}) == %View{
        changes: [],
        must_be_resolved?: false,
        size: 6,
        weight: 1,
        weighted_shape: [{2, 3}, {3, 1}]
      }
    end
  end

  describe "transpose/2" do
    test "adds changes" do
      view = View.build({2, 3})
      view_t = View.transpose(view, [0, 1])
      
      assert view.changes == []
      assert view_t.changes == [transpose: [0, 1]]
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
end
