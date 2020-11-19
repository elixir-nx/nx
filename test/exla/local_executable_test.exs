defmodule LocalExecutableTest do
  use ExUnit.Case
  alias Exla.Builder
  alias Exla.Client
  alias Exla.LocalExecutable
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Tensor

  setup_all do
    case System.fetch_env("EXLA_TARGET") do
      {:ok, "cuda"} ->
        {:ok, cpu: Client.create_client(), gpu: Client.create_client(platform: :cuda)}

      _ ->
        {:ok, cpu: Client.create_client(), gpu: nil}
    end
  end

  setup state do
    {:ok, builder: Builder.new("test"), cpu: state[:cpu], gpu: state[:gpu]}
  end

  test "run/4 succeeds with no inputs and default options on host device", state do
    # TODO: Not sure if this is the most efficient way to test all of this
    op = Op.constant(state[:builder], 1)
    comp = Builder.build(op)
    exec = Client.compile(state[:cpu], comp, {})
    assert %Tensor{data: {:ref, _}, shape: %Shape{}} = LocalExecutable.run(exec, {})
  end

  @tag :cuda
  test "run/4 succeeds with no inputs and default options on cuda device", state do
    # TODO: Not sure if this is the most efficient way to test all of this
    op = Op.constant(state[:builder], 1)
    comp = Builder.build(op)
    exec = Client.compile(state[:gpu], comp, {})

    assert %Tensor{data: {:ref, _}, shape: %Shape{}} =
             LocalExecutable.run(exec, {}, device: {:cuda, 0})
  end

  test "run/4 succeeds with 1 input and default options on host device", state do
    t1 = Tensor.scalar(1, :int32)
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    exec = Client.compile(state[:cpu], comp, {t1.shape})
    assert %Tensor{data: {:ref, _}, shape: %Shape{}} = LocalExecutable.run(exec, {t1})
  end

  @tag :cuda
  test "run/4 succeeds with 1 input and default options on cuda device", state do
    t1 = Tensor.scalar(1, :int32)
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    exec = Client.compile(state[:gpu], comp, {t1.shape})

    assert %Tensor{data: {:ref, _}, shape: %Shape{}} =
             LocalExecutable.run(exec, {t1}, device: {:cuda, 0})
  end

  test "run/4 succeeds with 2 inputs and default options on host device", state do
    t1 = Tensor.scalar(1, :int32)
    t2 = Tensor.scalar(1, :int32)
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    y = Op.parameter(state[:builder], 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(state[:cpu], comp, {t1.shape, t2.shape})
    assert %Tensor{data: {:ref, _}, shape: %Shape{}} = LocalExecutable.run(exec, {t1, t2})
  end

  @tag :cuda
  test "run/4 succeeds with 2 inputs and default options on cuda device", state do
    t1 = Tensor.scalar(1, :int32)
    t2 = Tensor.scalar(1, :int32)
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    y = Op.parameter(state[:builder], 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(state[:gpu], comp, {t1.shape, t2.shape})

    assert %Tensor{data: {:ref, _}, shape: %Shape{}} =
             LocalExecutable.run(exec, {t1, t2}, device: {:cuda, 0})
  end
end
