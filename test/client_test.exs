defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Op
  alias Exla.LocalExecutable
  alias Exla.Builder
  alias Exla.Shape

  # We need a common client for each test
  # TODO: As is, this will crash with the CPU-only build so we just have to handle that
  # and then we can effectively exclude cuda tagged tests
  setup_all do
    # Don't crash on a CPU-only build...still need to exclude cuda tests
    case System.fetch_env("EXLA_TARGET") do
      {:ok, "cuda"} -> {:ok, cpu: Client.create_client(), gpu: Client.create_client()}
      _ -> {:ok, cpu: Client.create_client(), gpu: nil}
    end
  end

  # We'll need a new builder before each test
  setup state do
    {:ok, cpu: state[:cpu], gpu: state[:gpu], builder: Builder.new("test")}
  end

  test "create_client/1 succeeds on host device with default args", state do
    assert %Client{} = state[:cpu]
  end

  @tag :cuda
  test "create_client/1 succeeds on cuda device with default args", state do
    assert %Client{} = state[:gpu]
  end

  test "get_default_device_ordinal/1 returns nonnegative integer on cpu", state do
    ordinal = Client.get_default_device_ordinal(state[:cpu])
    assert is_integer(ordinal)
    assert ordinal >= 0
  end

  @tag :cuda
  test "get_default_device_ordinal/1 returns nonnegative integer on gpu", state do
    ordinal = Client.get_default_device_ordinal(state[:gpu])
    assert is_integer(ordinal)
    assert ordinal >= 0
  end

  test "get_device_count/1 returns nonnegative integer on cpu", state do
    count = Client.get_device_count(state[:cpu])
    assert is_integer(count)
    assert count >= 0
  end

  @tag :cuda
  test "get_device_count/1 returns nonnegative integer on gpu", state do
    count = Client.get_device_count(state[:gpu])
    assert is_integer(count)
    assert count >= 0
  end

  test "compile/4 succeeds on host device with constant computation and no args", state do
    op = Op.constant(state[:builder], 1)
    comp = Builder.build(op)
    assert %LocalExecutable{} = Client.compile(state[:cpu], comp, {})
  end

  test "compile/4 succeeds on host device with basic computation and args", state do
    shape = Shape.make_shape(:int32, {})
    x = Op.parameter(state[:builder], 0, shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    assert %LocalExecutable{} = Client.compile(state[:cpu], comp, {shape})
  end
end
