defmodule OpTest do
  use ExUnit.Case
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Builder

  test "parameter/4 successfully creates op" do
    {:ok, builder = %Builder{}} = Builder.new("test")
    {:ok, shape = %Shape{}} = Shape.make_shape(:int32, {1, 1})
    assert {:ok, %Op{}} = Op.parameter(builder, 0, shape, "x")
  end

  test "constant/2 successfully creates constant op" do
    {:ok, builder = %Builder{}} = Builder.new("test")
    assert {:ok, %Op{}} = Op.constant(builder, 1)
  end
end
