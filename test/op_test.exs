defmodule OpTest do
  use ExUnit.Case
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Builder

  test "parameter/4 successfully creates op" do
    builder = Builder.new("test")
    shape = Shape.make_shape(:int32, {1, 1})
    assert %Op{} = Op.parameter(builder, 0, shape, "x")
  end

  test "constant/2 successfully creates constant op" do
    builder = Builder.new("test")
    assert %Op{} = Op.constant(builder, 1)
  end
end
