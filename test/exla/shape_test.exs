defmodule Exla.ShapeTest do
  use ExUnit.Case, async: true

  alias Exla.Shape

  describe "make_shape/2" do
    test "creates shape" do
      assert %Shape{dtype: {:s, 32}, dims: {1, 1}, ref: _} = Shape.make_shape({:s, 32}, {1, 1})
    end

    test "creates bf16 shape" do
      assert %Shape{dtype: {:bf, 16}, dims: {}, ref: _} = Shape.make_shape({:bf, 16}, {})
    end
  end

  test "make_tuple_shape/1" do
    s1 = Shape.make_shape({:s, 32}, {1, 1})
    s2 = Shape.make_shape({:s, 32}, {2, 2})

    assert %Shape{
             dtype:
               {:t,
                [%Shape{dtype: {:s, 32}, dims: {1, 1}}, %Shape{dtype: {:s, 32}, dims: {2, 2}}]},
             dims: {2}
           } = Shape.make_tuple_shape([s1, s2])
  end
end
