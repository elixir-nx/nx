defmodule ExecutableTest do
  use ExUnit.Case, async: true
  alias Exla.Builder
  alias Exla.Client
  alias Exla.Executable
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Buffer

  import ExlaHelpers

  setup do
    {:ok, builder: Builder.new("test")}
  end

  test "run/4 succeeds with no inputs and default options", config do
    # TODO: Not sure if this is the most efficient way to test all of this
    op = Op.constant(config.builder, 1)
    comp = Builder.build(op)
    exec = Client.compile(client(), comp, [])
    assert %Buffer{data: <<1, 0, 0, 0>>} = Executable.run(exec, [])
  end

  test "run/4 succeeds with 1 input and default options", config do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    exec = Client.compile(client(), comp, [t1.shape])
    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1])
  end

  test "run/4 succeeds with 2 inputs and default options", config do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    x = Op.parameter(config.builder, 0, t1.shape, "x")
    y = Op.parameter(config.builder, 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(client(), comp, [t1.shape, t2.shape])
    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1, t2])
  end
end
