defmodule EXLA.ShapeTest do
  use ExUnit.Case, async: true

  alias EXLA.Shape

  describe "make_shape/2" do
    test "creates shape" do
      shape = Shape.make_shape({:s, 32}, {1, 1})
      assert %Shape{dtype: {:s, 32}, dims: {1, 1}, ref: _} = shape
      assert Shape.byte_size(shape) == 4
    end

    test "creates bf16 shape" do
      shape = Shape.make_shape({:bf, 16}, {})
      assert %Shape{dtype: {:bf, 16}, dims: {}, ref: _} = shape
      assert Shape.byte_size(shape) == 2
    end
  end
end
