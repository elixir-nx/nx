defmodule ExecutableTest do
  use ExUnit.Case
  alias Exla.Builder
  alias Exla.Client
  alias Exla.Executable
  alias Exla.Op
  alias Exla.Shape
  alias Exla.Buffer

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
    op = Op.constant_r0(state[:builder], 1, {:s, 8})
    comp = Builder.build(op)
    exec = Client.compile(state[:cpu], comp, [])
    assert %Buffer{data: <<1>>} = Executable.run(exec, [])
  end

  @tag :cuda
  test "run/4 succeeds with no inputs and default options on cuda device", state do
    # TODO: Not sure if this is the most efficient way to test all of this
    op = Op.constant_r0(state[:builder], 1, {:s, 16})
    comp = Builder.build(op)
    exec = Client.compile(state[:gpu], comp, [])

    assert %Buffer{data: <<1, 0>>} = Executable.run(exec, [], device: {:cuda, 0})
  end

  test "run/4 succeeds with 1 input and default options on host device", state do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    exec = Client.compile(state[:cpu], comp, [t1.shape])
    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1])
  end

  @tag :cuda
  test "run/4 succeeds with 1 input and default options on cuda device", state do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    res = Op.add(x, x)
    comp = Builder.build(res)
    exec = Client.compile(state[:gpu], comp, [t1.shape])

    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1], device: {:cuda, 0})
  end

  test "run/4 succeeds with 2 inputs and default options on host device", state do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    y = Op.parameter(state[:builder], 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(state[:cpu], comp, [t1.shape, t2.shape])
    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1, t2])
  end

  test "slice", state do
    op = Op.constant_r1(state[:builder], 5, 1)
    op = Op.slice(op, [2], [4], [1])
    comp = Builder.build(op)
    exec = Client.compile(state[:cpu], comp, [])
    assert %Buffer{} = Executable.run(exec, [])
  end

  @tag :cuda
  test "run/4 succeeds with 2 inputs and default options on cuda device", state do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:i, 32}, {})}
    x = Op.parameter(state[:builder], 0, t1.shape, "x")
    y = Op.parameter(state[:builder], 1, t2.shape, "y")
    res = Op.add(x, y)
    comp = Builder.build(res)
    exec = Client.compile(state[:gpu], comp, [t1.shape, t2.shape])

    assert %Buffer{data: <<2, 0, 0, 0>>} = Executable.run(exec, [t1, t2], device: {:cuda, 0})
  end
end
