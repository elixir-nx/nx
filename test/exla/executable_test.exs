defmodule ExecutableTest do
  use ExUnit.Case, async: true

  alias Exla.{Buffer, Builder, Client, Executable, Op, Shape}

  import ExlaHelpers

  setup do
    {:ok, builder: Builder.new("test")}
  end

  test "run/4 succeeds with no inputs and default options", config do
    op = Op.constant_r0(config.builder, 1, {:s, 32})
    comp = Builder.build(op)
    exec = Client.compile(client(), comp, [])
    assert %Buffer{data: <<1, 0, 0, 0>>} = Executable.run(exec, [])
  end

  test "run/4 succeeds with 1 input and default options", config do
    t1 = %Buffer{data: <<1::8-native>>, shape: Shape.make_shape({:s, 8}, {})}
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    comp = Builder.build(x)
    exec = Client.compile(client(), comp, [t1.shape])
    assert %Buffer{data: <<1>>} = Executable.run(exec, [t1])
  end

  test "run/4 succeeds with 2 inputs and default options", config do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    y = Op.parameter(config.builder, 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(client(), comp, [t1.shape, t2.shape])
    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1, t2])
  end

  test "run/4 returns a ref when keep_on_device is true", config do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    y = Op.parameter(config.builder, 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(client(), comp, [t1.shape, t2.shape])
    assert %Buffer{ref: {ref, _, _}} = Executable.run(exec, [t1, t2], keep_on_device: true)
    assert is_reference(ref)
  end

  test "run/4 succeeds when data is pre-loaded", config do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t1 = Buffer.place_on_device(client(), t1, {client().platform, 0})
    t2 = Buffer.place_on_device(client(), t2, {client().platform, 0})
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    y = Op.parameter(config.builder, 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(client(), comp, [t1.shape, t2.shape])
    assert %Buffer{ref: {ref, _, _}} = Executable.run(exec, [t1, t2], keep_on_device: true)
    assert is_reference(ref)
  end

  test "run/4 succeeds with data from a previous run", config do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    y = Op.parameter(config.builder, 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(client(), comp, [t1.shape, t2.shape])
    assert t3 = %Buffer{ref: ref} = Executable.run(exec, [t1, t2], keep_on_device: true)
    assert %Buffer{data: <<4::32-native>>} = Executable.run(exec, [t3, t3])
  end
end
