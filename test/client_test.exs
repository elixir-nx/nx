defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Op
  alias Exla.LocalExecutable
  alias Exla.Builder
  alias Exla.Shape

  # We need a common client for each test
  setup_all do
    {:ok, client: Client.create_client()}
  end

  # We'll need a new builder before each test
  setup state do
    {:ok, client: state[:client], builder: Builder.new("test")}
  end

  test "create_client/1 succeeds on host device with default args", state do
    assert %Client{} = state[:client]
  end

  test "compile/4 succeeds on host device with constant computation and no args", state do
    op = Op.constant(state[:builder], 1)
    comp = Builder.build(op)
    assert %LocalExecutable{} = Client.compile(state[:client], comp, {})
  end

  test "compile/4 succeeds on host device with basic computation and args", state do
    shape = Shape.make_shape(:int32, {})
    x = Op.parameter(state[:builder], 0, shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    assert %LocalExecutable{} = Client.compile(state[:client], comp, {shape})
  end
end
