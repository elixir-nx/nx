defmodule OpTest do
  use ExUnit.Case
  alias Exla.Op
  alias Exla.Shape

  test "parameter/3 successfully creates op" do
    {:ok, shape = %Shape{}} = Exla.Shape.make_shape(:int32, {1, 1})
    assert {:ok, %Op{}} = Exla.Op.parameter(0, shape, "x")
  end
end