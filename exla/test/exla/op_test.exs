defmodule EXLA.OpTest do
  use ExUnit.Case, async: true

  alias EXLA.{Builder, Shape, Op}

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
    assert d = %Op{} = Op.constant_r0(builder, -100_000, {:s, 64})
    assert e = %Op{} = Op.constant_r0(builder, -1000, {:s, 16})
    assert f = %Op{} = Op.constant_r0(builder, -100, {:s, 8})
    assert g = %Op{} = Op.constant_r0(builder, 100, {:u, 8})
    assert h = %Op{} = Op.constant_r0(builder, 1000, {:u, 16})
    assert i = %Op{} = Op.constant_r0(builder, 10000, {:u, 32})
    assert j = %Op{} = Op.constant_r0(builder, 100_000, {:u, 64})

    assert %Shape{dims: {}, dtype: {:f, 64}} = Op.get_shape(a)
    assert %Shape{dims: {}, dtype: {:f, 32}} = Op.get_shape(b)
    assert %Shape{dims: {}, dtype: {:s, 32}} = Op.get_shape(c)
    assert %Shape{dims: {}, dtype: {:s, 64}} = Op.get_shape(d)
    assert %Shape{dims: {}, dtype: {:s, 16}} = Op.get_shape(e)
    assert %Shape{dims: {}, dtype: {:s, 8}} = Op.get_shape(f)
    assert %Shape{dims: {}, dtype: {:u, 8}} = Op.get_shape(g)
    assert %Shape{dims: {}, dtype: {:u, 16}} = Op.get_shape(h)
    assert %Shape{dims: {}, dtype: {:u, 32}} = Op.get_shape(i)
    assert %Shape{dims: {}, dtype: {:u, 64}} = Op.get_shape(j)
  end

  test "constant_from_binary/3" do
    builder = Builder.new("test")

    shape = Shape.make_shape({:s, 64}, {0})
    assert a = %Op{} = Op.constant_from_binary(builder, <<>>, shape)

    shape = Shape.make_shape({:s, 64}, {})
    assert b = %Op{} = Op.constant_from_binary(builder, <<1::64-native>>, shape)

    shape = Shape.make_shape({:s, 8}, {1, 4})
    assert c = %Op{} = Op.constant_from_binary(builder, <<1, 2, 3, 4>>, shape)

    shape = Shape.make_shape({:s, 16}, {1, 1, 1, 1})
    assert d = %Op{} = Op.constant_from_binary(builder, <<1::16>>, shape)

    assert e =
             %Op{} =
             Op.constant_from_binary(
               builder,
               <<1::float-native, 2::float-native, 3::float-native, 4::float-native>>,
               Shape.make_shape({:f, 64}, {4, 1})
             )

    assert %Shape{dims: {0}, dtype: {:s, 64}} = Op.get_shape(a)
    assert %Shape{dims: {}, dtype: {:s, 64}} = Op.get_shape(b)
    assert %Shape{dims: {1, 4}, dtype: {:s, 8}} = Op.get_shape(c)
    assert %Shape{dims: {1, 1, 1, 1}, dtype: {:s, 16}} = Op.get_shape(d)
    assert %Shape{dims: {4, 1}, dtype: {:f, 64}} = Op.get_shape(e)

    assert_raise ArgumentError, "binary does not match the given type and dimensions", fn ->
      shape = Shape.make_shape({:s, 64}, {2, 2})
      Op.constant_from_binary(builder, <<1::64-native>>, shape)
    end
  end

  test "tuple/2" do
    builder = Builder.new("test")

    shape_a = Shape.make_shape({:s, 64}, {0})
    a = Op.constant_from_binary(builder, <<>>, shape_a)
    shape_b = Shape.make_shape({:s, 64}, {})
    b = Op.constant_from_binary(builder, <<1::64-native>>, shape_b)
    shape_c = Shape.make_shape({:s, 8}, {1, 4})
    c = Op.constant_from_binary(builder, <<1, 2, 3, 4>>, shape_c)
    shape_d = Shape.make_shape({:s, 16}, {1, 1, 1, 1})
    d = Op.constant_from_binary(builder, <<1::16>>, shape_d)

    assert e = %Op{} = Op.tuple(builder, [a, b, c, d])
    assert f = %Op{} = Op.tuple(builder, [e, a])
    assert g = %Op{} = Op.tuple(builder, [])

    assert h =
             %Op{} =
             Op.tuple(builder, [Op.tuple(builder, [e, g]), b, c, f, Op.tuple(builder, [d])])

    assert %Shape{
             dims: {4},
             dtype:
               {:tuple,
                [
                  %Shape{dtype: {:s, 64}, dims: {0}},
                  %Shape{dtype: {:s, 64}, dims: {}},
                  %Shape{dtype: {:s, 8}, dims: {1, 4}},
                  %Shape{dtype: {:s, 16}, dims: {1, 1, 1, 1}}
                ]}
           } = Op.get_shape(e)

    assert %Shape{
             dims: {2},
             dtype:
               {:tuple,
                [
                  %Shape{
                    dims: {4},
                    dtype:
                      {:tuple,
                       [
                         %Shape{dtype: {:s, 64}, dims: {0}},
                         %Shape{dtype: {:s, 64}, dims: {}},
                         %Shape{dtype: {:s, 8}, dims: {1, 4}},
                         %Shape{dtype: {:s, 16}, dims: {1, 1, 1, 1}}
                       ]}
                  },
                  %Shape{dtype: {:s, 64}, dims: {0}}
                ]}
           } = Op.get_shape(f)

    assert %Shape{
             dims: {5},
             dtype:
               {:tuple,
                [
                  %Shape{
                    dims: {2},
                    dtype:
                      {:tuple,
                       [
                         %Shape{
                           dims: {4},
                           dtype:
                             {:tuple,
                              [
                                %Shape{dtype: {:s, 64}, dims: {0}},
                                %Shape{dtype: {:s, 64}, dims: {}},
                                %Shape{dtype: {:s, 8}, dims: {1, 4}},
                                %Shape{dtype: {:s, 16}, dims: {1, 1, 1, 1}}
                              ]}
                         },
                         %Shape{dims: {0}, dtype: {:tuple, []}}
                       ]}
                  },
                  %Shape{dims: {}, dtype: {:s, 64}},
                  %Shape{dims: {1, 4}, dtype: {:s, 8}},
                  %Shape{
                    dims: {2},
                    dtype:
                      {:tuple,
                       [
                         %Shape{
                           dims: {4},
                           dtype:
                             {:tuple,
                              [
                                %Shape{dtype: {:s, 64}, dims: {0}},
                                %Shape{dtype: {:s, 64}, dims: {}},
                                %Shape{dtype: {:s, 8}, dims: {1, 4}},
                                %Shape{dtype: {:s, 16}, dims: {1, 1, 1, 1}}
                              ]}
                         },
                         %Shape{dtype: {:s, 64}, dims: {0}}
                       ]}
                  },
                  %Shape{
                    dims: {1},
                    dtype: {:tuple, [%Shape{dims: {1, 1, 1, 1}, dtype: {:s, 16}}]}
                  }
                ]}
           } = Op.get_shape(h)

    assert %Shape{dims: {0}, dtype: {:tuple, []}} = Op.get_shape(g)
  end

  test "get_tuple_element/2" do
    builder = Builder.new("test")

    shape_a = Shape.make_shape({:s, 64}, {0})
    a = Op.constant_from_binary(builder, <<>>, shape_a)
    shape_b = Shape.make_shape({:s, 64}, {})
    b = Op.constant_from_binary(builder, <<1::64-native>>, shape_b)
    shape_c = Shape.make_shape({:s, 8}, {1, 4})
    c = Op.constant_from_binary(builder, <<1, 2, 3, 4>>, shape_c)
    shape_d = Shape.make_shape({:s, 16}, {1, 1, 1, 1})
    d = Op.constant_from_binary(builder, <<1::16>>, shape_d)

    e = Op.tuple(builder, [a, b, c, d])
    f = Op.tuple(builder, [e, c])

    assert g = %Op{} = Op.get_tuple_element(e, 3)
    assert h = %Op{} = Op.get_tuple_element(f, 0)

    assert %Shape{dtype: {:s, 16}, dims: {1, 1, 1, 1}} = Op.get_shape(g)

    assert %Shape{
             dims: {4},
             dtype:
               {:tuple,
                [
                  %Shape{dtype: {:s, 64}, dims: {0}},
                  %Shape{dtype: {:s, 64}, dims: {}},
                  %Shape{dtype: {:s, 8}, dims: {1, 4}},
                  %Shape{dtype: {:s, 16}, dims: {1, 1, 1, 1}}
                ]}
           } = Op.get_shape(h)
  end

  test "add/3 successfully creates add op without broadcast dimensions" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    a = Op.parameter(builder, 0, shape, "a")
    b = Op.parameter(builder, 1, shape, "b")
    assert %Op{} = Op.add(a, b)
  end

  test "dot/3 successfully creates dot op" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:s, 32}, {1, 1})
    a = Op.parameter(builder, 0, shape, "a")
    b = Op.parameter(builder, 1, shape, "b")
    assert %Op{} = Op.dot(a, b, :high)
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

  test "get_shape/1 returns shape of op" do
    builder = Builder.new("test")
    shape = Shape.make_shape({:f, 64}, {5, 5, 5, 5, 5})
    x = Op.parameter(builder, 0, shape, "x")
    assert %Shape{dims: {5, 5, 5, 5, 5}, dtype: {:f, 64}} = Op.get_shape(x)
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

    assert %Shape{dims: {1, 1}, dtype: {:f, 32}} = Op.get_shape(y)
    assert %Shape{dims: {1, 1}, dtype: {:f, 16}} = Op.get_shape(z)
    assert %Shape{dims: {1, 1}, dtype: {:s, 8}} = Op.get_shape(a)
    assert %Shape{dims: {1, 1}, dtype: {:bf, 16}} = Op.get_shape(b)
    assert %Shape{dims: {1, 1}, dtype: {:u, 16}} = Op.get_shape(c)
    assert %Shape{dims: {1, 1}, dtype: {:c, 64}} = Op.get_shape(d)
    assert %Shape{dims: {1, 1}, dtype: {:c, 128}} = Op.get_shape(e)

    assert_raise RuntimeError,
                 "Conversion from complex to real type c128[1,1] => S32 is not implemented.",
                 fn -> Op.get_shape(Op.convert_element_type(e, {:s, 32})) end
  end

  test "rng_normal/3" do
    builder = Builder.new("test")

    shape_a = Shape.make_shape({:f, 64}, {3, 3, 3})
    mu_a = Op.constant_r0(builder, 0, {:f, 64})
    sigma_a = Op.constant_r0(builder, 1, {:f, 64})
    a = Op.rng_normal(mu_a, sigma_a, shape_a)

    shape_b = Shape.make_shape({:f, 32}, {2, 2, 2})
    mu_b = Op.constant_r0(builder, 0, {:f, 32})
    sigma_b = Op.constant_r0(builder, 1, {:f, 32})
    b = Op.rng_normal(mu_b, sigma_b, shape_b)

    assert %Shape{dims: {3, 3, 3}, dtype: {:f, 64}} = Op.get_shape(a)
    assert %Shape{dims: {2, 2, 2}, dtype: {:f, 32}} = Op.get_shape(b)
  end

  test "rng_uniform/3" do
    builder = Builder.new("test")

    shape_a = Shape.make_shape({:f, 64}, {3, 3, 3})
    low_a = Op.constant_r0(builder, 0, {:f, 64})
    high_a = Op.constant_r0(builder, 1, {:f, 64})
    a = Op.rng_uniform(low_a, high_a, shape_a)

    shape_b = Shape.make_shape({:f, 32}, {3, 3, 3})
    low_b = Op.constant_r0(builder, 0, {:f, 32})
    high_b = Op.constant_r0(builder, 1, {:f, 32})
    b = Op.rng_uniform(low_b, high_b, shape_b)

    shape_c = Shape.make_shape({:s, 32}, {2, 2, 2})
    low_c = Op.constant_r0(builder, 0, {:s, 32})
    high_c = Op.constant_r0(builder, 5, {:s, 32})
    c = Op.rng_uniform(low_c, high_c, shape_c)

    assert %Shape{dims: {3, 3, 3}, dtype: {:f, 64}} = Op.get_shape(a)
    assert %Shape{dims: {3, 3, 3}, dtype: {:f, 32}} = Op.get_shape(b)
    assert %Shape{dims: {2, 2, 2}, dtype: {:s, 32}} = Op.get_shape(c)
  end
end
