defmodule Exla.ShapeTest do
  use ExUnit.Case, async: true

  alias Exla.{Builder, Op, Shape}

  describe "make_shape/2" do
    test "creates shape" do
      assert %Shape{dtype: {:s, 32}, dims: {1, 1}, ref: _} = Shape.make_shape({:s, 32}, {1, 1})
    end

    test "creates bf16 shape" do
      assert %Shape{dtype: {:bf, 16}, dims: {}, ref: _} = Shape.make_shape({:bf, 16}, {})
    end
  end

  describe "get_shape/1" do
    test "returns shape of op" do
      builder = Builder.new("test")
      shape = Shape.make_shape({:f, 64}, {5, 5, 5, 5, 5})
      x = Op.parameter(builder, 0, shape, "x")
      assert %Shape{dims: {5, 5, 5, 5, 5}, dtype: {:f, 64}} = Shape.get_shape(x)
    end
  end
end
