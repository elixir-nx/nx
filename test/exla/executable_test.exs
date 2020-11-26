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
end
