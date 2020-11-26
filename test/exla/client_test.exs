defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Op
  alias Exla.Executable
  alias Exla.Builder
  alias Exla.Shape

  # We need a common client for each test
  setup_all do
    # TODO: This needs to be moved, but for now it's the only way to get the CUDA tests to pass
    # because compilation relies on the TF SubProcess class which needs `waitpid` to behave well
    # to work.
    :os.set_signal(:sigchld, :default)

    case System.fetch_env("EXLA_TARGET") do
      {:ok, "cuda"} ->
        {:ok, cpu: Client.create_client(), gpu: Client.create_client(platform: :cuda)}

      _ ->
        {:ok, cpu: Client.create_client(), gpu: nil}
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

  test "compile/4 succeeds on host device with constant_r0 computation and no args", state do
    op = Op.constant_r0(state[:builder], 1, {:s, 32})
    comp = Builder.build(op)
    assert %Executable{} = Client.compile(state[:cpu], comp, [])
  end

  @tag :cuda
  test "compile/4 succeeds on cuda device with constant_r0 computation and no args", state do
    op = Op.constant_r0(state[:builder], 1, {:s, 32})
    comp = Builder.build(op)
    assert %Executable{} = Client.compile(state[:cpu], comp, [])
  end

  test "compile/4 succeeds on host device with basic computation and args", state do
    shape = Shape.make_shape({:i, 32}, {})
    x = Op.parameter(state[:builder], 0, shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    assert %Executable{} = Client.compile(state[:cpu], comp, [shape])
  end

  @tag :cuda
  test "compile/4 succeeds on cuda device with basic computation and args", state do
    shape = Shape.make_shape({:i, 32}, {})
    x = Op.parameter(state[:builder], 0, shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    assert %Executable{} = Client.compile(state[:gpu], comp, [shape])
  end
end
