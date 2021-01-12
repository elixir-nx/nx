defmodule EXLA.ShapeTest do
  use ExUnit.Case, async: true

  alias EXLA.Shape

  describe "make_shape/2" do
    test "creates shape" do
      assert %Shape{dtype: {:s, 32}, dims: {1, 1}, ref: _} = Shape.make_shape({:s, 32}, {1, 1})
    end

    test "creates bf16 shape" do
      assert %Shape{dtype: {:bf, 16}, dims: {}, ref: _} = Shape.make_shape({:bf, 16}, {})
    end
  end
end
