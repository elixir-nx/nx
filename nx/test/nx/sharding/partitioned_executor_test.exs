defmodule Nx.Sharding.PartitionedExecutorTest do
  use ExUnit.Case, async: false

  @moduletag :distributed

  alias Nx.Sharding.PartitionedExecutor.Function, as: F

  test "executes a graph in a single node" do
    # Manually creating a graph of:application
    # f(x, y) = {x + y, x - y}
    # g(z) = {0, z - 1}
    # h(f, g) = {f[0], f[1], g[1]}

    x = :rand.uniform()
    y = 10 + :rand.uniform()
    z = 100 + :rand.uniform()

    test_pid = self()
    ref = make_ref()

    # Note: these names only work in this test because we know that the test isn't concurrent with anything.
    # In normal usage, the names should be refs created via make_ref or something similar.
    graph = [
      %F{id: :x, args: [], code: nil, results: [x], node: nil},
      %F{id: :y, args: [], code: nil, results: [y], node: nil},
      %F{id: :z, args: [], code: nil, results: [z], node: nil},
      %F{
        id: :f,
        args: [{:x, 0}, {:y, 0}],
        code: &TestFunctions.f/1,
        results: nil,
        node: nil
      },
      %F{id: :g, args: [{:z, 0}], code: &TestFunctions.g/1, results: nil, node: nil},
      %F{
        id: :h,
        args: [{:f, 0}, {:f, 1}, {:g, 1}],
        code: &TestFunctions.h/1,
        results: nil,
        node: nil
      },
      %F{
        id: :output,
        args: [{:h, 0}, {:h, 1}, {:h, 2}],
        code: fn output ->
          send(test_pid, {:result, ref, output})
          []
        end,
        results: nil,
        node: nil
      }
    ]

    {:ok, _executor} = Nx.Sharding.PartitionedExecutor.start_link(graph)

    assert_receive {:result, ^ref, result}
    assert result == [x + y, x - y, z - 1]
  end

  test "executes a graph in multiple nodes" do
    # Note: this is the same graph as the previous test
    # but it runs some graph nodes on different BEAM nodes

    # Manually creating a graph of:application
    # f(x, y) = {x + y, x - y}
    # g(z) = {0, z - 1}
    # h(f, g) = {f[0], f[1], g[1]}

    x = :rand.uniform()
    y = 10 + :rand.uniform()
    z = 100 + :rand.uniform()

    test_pid = self()
    ref = make_ref()

    primary_node = Node.self()
    secondary_node = :"secondary@127.0.0.1"

    # Note: these names only work in this test because we know that the test isn't concurrent with anything.
    # In normal usage, the names should be refs created via make_ref or something similar.
    graph = [
      %F{id: :x, args: [], code: nil, results: [x], node: nil},
      %F{id: :y, args: [], code: nil, results: [y], node: secondary_node},
      %F{id: :z, args: [], code: nil, results: [z], node: secondary_node},
      %F{
        id: :f,
        args: [{:x, 0}, {:y, 0}],
        code: &TestFunctions.f/1,
        results: nil,
        node: primary_node
      },
      %F{
        id: :g,
        args: [{:z, 0}],
        code: &TestFunctions.g/1,
        results: nil,
        node: secondary_node
      },
      %F{
        id: :h,
        args: [{:f, 0}, {:f, 1}, {:g, 1}],
        code: &TestFunctions.h/1,
        results: nil,
        node: secondary_node
      },
      %F{
        id: :output,
        args: [{:h, 0}, {:h, 1}, {:h, 2}],
        code: fn output ->
          send(test_pid, {:result, ref, output})
          []
        end,
        results: nil,
        node: nil
      }
    ]

    {:ok, _executor} = Nx.Sharding.PartitionedExecutor.start_link(graph)

    assert_receive {:result, ^ref, result}, 5_000
    assert result == [x + y, x - y, z - 1]
  end
end
