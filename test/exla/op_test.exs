defmodule Exla.OpTest do
  use ExUnit.Case, async: true

  alias Exla.{Builder, Shape, Op}

  test "parameter/4 successfully creates op" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    assert %Op{} = Op.parameter(builder, 0, shape, "x")
  end

  test "constant_r0/3 successfully creates constant op" do
    builder = Builder.new("test")
    assert a = %Op{} = Op.constant_r0(builder, 1.0, {:f, 64})
    assert b = %Op{} = Op.constant_r0(builder, 1.0, {:f, 32})
    assert c = %Op{} = Op.constant_r0(builder, -10000, {:s, 32})
    assert d = %Op{} = Op.constant_r0(builder, -100000, {:s, 64})
    assert e = %Op{} = Op.constant_r0(builder, -1000, {:s, 16})
    assert f = %Op{} = Op.constant_r0(builder, -100, {:s, 8})
    assert g = %Op{} = Op.constant_r0(builder, 100, {:u, 8})
    assert h = %Op{} = Op.constant_r0(builder, 1000, {:u, 16})
    assert i = %Op{} = Op.constant_r0(builder, 10000, {:u, 32})
    assert j = %Op{} = Op.constant_r0(builder, 100000, {:u, 64})

    assert %Shape{dims: {}, dtype: {:f, 64}} = Shape.get_shape(a)
    assert %Shape{dims: {}, dtype: {:f, 32}} = Shape.get_shape(b)
    assert %Shape{dims: {}, dtype: {:s, 32}} = Shape.get_shape(c)
    assert %Shape{dims: {}, dtype: {:s, 64}} = Shape.get_shape(d)
    assert %Shape{dims: {}, dtype: {:s, 16}} = Shape.get_shape(e)
    assert %Shape{dims: {}, dtype: {:s, 8}} = Shape.get_shape(f)
    assert %Shape{dims: {}, dtype: {:u, 8}} = Shape.get_shape(g)
    assert %Shape{dims: {}, dtype: {:u, 16}} = Shape.get_shape(h)
    assert %Shape{dims: {}, dtype: {:u, 32}} = Shape.get_shape(i)
    assert %Shape{dims: {}, dtype: {:u, 64}} = Shape.get_shape(j)

    assert_raise RuntimeError, "unsupported constant type.", fn ->
      Shape.get_shape(Op.constant_r0(builder, 1.0, {:f, 16}))
    end
        assert_raise RuntimeError, "unsupported constant type.", fn ->
      Shape.get_shape(Op.constant_r0(builder, 1.0, {:bf, 16}))
    end

    assert_raise RuntimeError, "unsupported constant type.", fn ->
      Shape.get_shape(Op.constant_r0(builder, 1.0, {:c, 64}))
    end

    assert_raise RuntimeError, "unsupported constant type.", fn ->
      Shape.get_shape(Op.constant_r0(builder, 1.0, {:c, 128}))
    end
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
    init_value = Op.constant_r0(builder, 0.0, {:f, 64})

    a = Op.parameter(sub_builder, 0, shape, "a")
    b = Op.parameter(sub_builder, 1, shape, "b")
    reduction_ast = Op.add(a, b)
    reduction = Builder.build(reduction_ast)

    reduction_dimension = {0}

    assert %Op{} = Op.reduce(operand, init_value, reduction, reduction_dimension)
  end

  test "convert_element_type/1 changes type of operand" do
    builder = Builder.new("test")

    shape = Shape.make_shape({:f, 64}, {1, 1})

    x = Op.parameter(builder, 0, shape, "x")
    y = Op.convert_element_type(x, {:f, 32})
    z = Op.convert_element_type(y, {:f, 16})
    a = Op.convert_element_type(z, {:s, 8})
    b = Op.convert_element_type(a, {:bf, 16})
    c = Op.convert_element_type(b, {:u, 16})
    d = Op.convert_element_type(c, {:c, 64})
    e = Op.convert_element_type(d, {:c, 128})

    assert %Shape{dims: {1, 1}, dtype: {:f, 32}} = Shape.get_shape(y)
    assert %Shape{dims: {1, 1}, dtype: {:f, 16}} = Shape.get_shape(z)
    assert %Shape{dims: {1, 1}, dtype: {:s, 8}} = Shape.get_shape(a)
    assert %Shape{dims: {1, 1}, dtype: {:bf, 16}} = Shape.get_shape(b)
    assert %Shape{dims: {1, 1}, dtype: {:u, 16}} = Shape.get_shape(c)
    assert %Shape{dims: {1, 1}, dtype: {:c, 64}} = Shape.get_shape(d)
    assert %Shape{dims: {1, 1}, dtype: {:c, 128}} = Shape.get_shape(e)

    assert_raise RuntimeError,
                 "Conversion from complex to real type c128[1,1] => S32 is not implemented.",
                 fn -> Shape.get_shape(Op.convert_element_type(e, {:s, 32})) end
  end
end
