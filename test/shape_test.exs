defmodule ShapeTest do
  use ExUnit.Case
  alias Exla.Shape

  test "make_shape/2 successfully creates shape" do
    assert {:ok, %Shape{dtype: :int32, dims: {1, 1}, ref: _}} = Shape.make_shape(:int32, {1, 1})
  end
end
