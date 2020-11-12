defmodule LocalExecutableTest do
  use ExUnit.Case
  alias Exla.Builder
  alias Exla.Client
  alias Exla.LocalExecutable
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Tensor

  setup_all do
    {:ok, client: Client.create_client()}
  end

  setup state do
    {:ok, client: state[:client], builder: Builder.new("test")}
  end

  test "run/4 succeeds with no inputs and default options", state do
    # TODO: Not sure if this is the most efficient way to test all of this
    op = Op.constant(state[:builder], 1)
    comp = Builder.build(op)
    exec = Client.compile(state[:client], comp, {})
    assert %Tensor{data: {:ref, _}, shape: %Shape{}} = LocalExecutable.run(state[:client], exec, {})
  end

  test "run/4 succeeds with 1 input and default options", state do
    t1 = Tensor.scalar(1, :int32)
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    exec = Client.compile(state[:client], comp, {t1.shape})
    assert %Tensor{data: {:ref, _}, shape: %Shape{}} = LocalExecutable.run(state[:client], exec, {t1})
  end

  test "run/4 succeeds with 2 inputs and default options", state do
    t1 = Tensor.scalar(1, :int32)
    t2 = Tensor.scalar(1, :int32)
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    y = Op.parameter(state[:builder], 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(state[:client], comp, {t1.shape, t2.shape})
    assert %Tensor{data: {:ref, _}, shape: %Shape{}} = LocalExecutable.run(state[:client], exec, {t1, t2})
  end
end
