defmodule ClientTest do
  use ExUnit.Case, async: true

  alias Exla.Client
  alias Exla.Op
  alias Exla.Executable
  alias Exla.Builder
  alias Exla.Shape

  import ExlaHelpers

  setup do
    {:ok, builder: Builder.new("test")}
  end

  test "get_default_device_ordinal/1 returns nonnegative integer", _config do
    ordinal = Client.get_default_device_ordinal(client())
    assert is_integer(ordinal)
    assert ordinal >= 0
  end

  test "get_device_count/1 returns nonnegative integer", _config do
    count = Client.get_device_count(client())
    assert is_integer(count)
    assert count >= 0
  end

  test "compile/4 succeeds with constant computation and no args", config do
    op = Op.constant(config.builder, 1)
    comp = Builder.build(op)
    assert %Executable{} = Client.compile(client(), comp, [])
  end

  test "compile/4 succeeds with basic computation and args", config do
    shape = Shape.make_shape({:i, 32}, {})
    x = Op.parameter(config.builder, 0, shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    assert %Executable{} = Client.compile(client(), comp, [shape])
  end
end
