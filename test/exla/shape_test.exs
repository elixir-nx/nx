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

  describe "make_tuple_shape/1" do
    test "creates tuple shape" do
      s1 = Shape.make_shape({:s, 32}, {5, 5, 5})
      s2 = Shape.make_shape({:bf, 16}, {})
      s3 = Shape.make_shape({:f, 32}, {1, 1})

      assert %Shape{dtype: {:t, [_, _, _]}, dims: {3}, ref: ref} = Shape.make_tuple_shape([s1, s2, s3])
      assert %Shape{dtype: {:t, [_, _, _]}, dims: {3}, ref: ref} = Shape.get_shape_info(ref)
    end

    test "creates nested tuples" do
      s1 = Shape.make_shape({:s, 32}, {5, 5, 5})
      s2 = Shape.make_shape({:bf, 16}, {})
      s3 = Shape.make_shape({:f, 32}, {1, 1})
      s4 = Shape.make_shape({:s, 32}, {1})
      t1 = Shape.make_tuple_shape([s1, s2, s3])

      assert %Shape{dtype: {:t, [_, %Shape{dtype: {:t, [_, _, _]}}]}, dims: {2}, ref: ref} = Shape.make_tuple_shape([s4, t1])
      assert %Shape{dtype: {:t, [_, %Shape{dtype: {:t, [_, _, _]}}]}, dims: {2}, ref: ref} = Shape.get_shape_info(ref)
    end
  end
end
