defmodule OpTest do
  use ExUnit.Case
  alias Exla.Op
  alias Exla.Shape

  test "parameter/3 successfully creates op" do
    {:ok, shape = %Shape{}} = Shape.make_shape(:int32, {1, 1})
    assert {:ok, %Op{}} = Op.parameter(0, shape, "x")
  end

  test "constant/1 successfully creates constant op" do
    assert {:ok, %Op{}} = Op.constant(1)
  end
end
