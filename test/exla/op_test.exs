defmodule OpTest do
  use ExUnit.Case
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Builder

  test "parameter/4 successfully creates op" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    assert %Op{} = Op.parameter(builder, 0, shape, "x")
  end

  test "constant/2 successfully creates constant op" do
    builder = Builder.new("test")
    assert %Op{} = Op.constant(builder, 1)
  end

  test "zero/2 successfully creates zero op" do
    builder = Builder.new("test")
    assert %Op{} = Op.zero(builder, {:f, 64})
  end

  test "add/3 successfully creates add op without broadcast dimensions" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    a = Op.parameter(builder, 0, shape, "a")
    b = Op.parameter(builder, 1, shape, "b")
    assert %Op{} = Op.add(a, b)
  end

  test "div/3 successfully creates div op without broadcast dimensions" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    a = Op.parameter(builder, 0, shape, "a")
    b = Op.parameter(builder, 1, shape, "b")
    assert %Op{} = Op.div(a, b)
  end

  test "dot/2 successfully creates dot op" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    a = Op.parameter(builder, 0, shape, "a")
    b = Op.parameter(builder, 1, shape, "b")
    assert %Op{} = Op.dot(a, b)
  end

  test "exp/1 successfully creates exp op" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    a = Op.parameter(builder, 0, shape, "a")
    assert %Op{} = Op.exp(a)
  end

  test "reduce/4 successfully creates reduce op" do
    builder = Builder.new("test")
    sub_builder = Builder.new(builder, "sub_test")

    shape = Shape.make_shape({:f, 64}, {1_000})
    operand = Op.parameter(builder, 0, shape, "x")
    init_value = Op.constant(builder, 0)

    a = Op.parameter(sub_builder, 0, shape, "a")
    b = Op.parameter(sub_builder, 1, shape, "b")
    reduction_ast = Op.add(a, b)
    reduction = Builder.build(reduction_ast)

    reduction_dimension = {0}

    assert %Op{} = Op.reduce(operand, init_value, reduction, reduction_dimension)
  end

  test "get_shape/1 returns shape of op" do
    builder = Builder.new("test")

    shape = Shape.make_shape({:f, 64}, {5, 5, 5, 5, 5})

    x = Op.parameter(builder, 0, shape, "x")
    assert %Shape{dims: {5, 5, 5, 5, 5}, dtype: {:f, 64}} = Op.get_shape(x)
  end
end
