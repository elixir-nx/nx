defmodule Nx.Defn.ShardJitTest do
  use ExUnit.Case, async: true

  # A minimal Nx.Defn.Compiler that just records the arguments it receives
  # from Nx.Defn.Compiler.__shard_jit__/5, so we can assert on how
  # `vars` and `options[:input_shardings]` were flattened/validated before
  # ever reaching a backend. Real backends (e.g. EXLA) receive the exact
  # same shapes this test asserts on.
  defmodule TestCompiler do
    @behaviour Nx.Defn.Compiler

    @impl true
    def __to_backend__(_opts), do: {Nx.BinaryBackend, []}

    @impl true
    def __partitions_options__(opts), do: [opts]

    @impl true
    def __compile__(_key, _vars, _fun, _opts) do
      raise "not implemented"
    end

    @impl true
    def __jit__(_key, _vars, fun, args_list, _opts) do
      Enum.map(args_list, &apply(fun, &1))
    end

    @impl true
    def __shard_jit__(_key, mesh, vars, _fun, args_list, opts) do
      send(self(), {:shard_jit, mesh, vars, Keyword.fetch!(opts, :input_shardings)})
      Enum.map(args_list, fn _ -> :ok end)
    end
  end

  describe "container arguments" do
    test "expands a single sharding spec across every leaf of a container argument" do
      mesh = %Nx.Mesh{name: "mesh", shape: {2}}
      fun = fn container, tensor -> Nx.add(container.a, tensor) end

      # `container` has two leaves (a, b); `tensor` has one. A single "%{}"
      # (fully replicated) spec on the container argument must apply to
      # both of its leaves.
      container = %{a: Nx.tensor([1, 2]), b: Nx.tensor([10, 20])}
      tensor = Nx.tensor([100])

      args_list = [[container, tensor], [container, tensor]]

      Nx.Defn.shard_jit(fun, mesh,
        compiler: TestCompiler,
        input_shardings: [%{}, %{0 => [0]}]
      ).(args_list)

      assert_received {:shard_jit, ^mesh, [container_var, tensor_var], input_shardings}

      # One list-format entry per leaf: container.a, container.b, tensor.
      assert input_shardings == [[[]], [[]], [[0]]]

      # container.a/b stay fully replicated (unsharded shape == given shape),
      # while tensor's unsharded shape is scaled up by the mesh axis it uses.
      assert container_var.a.shape == {2}
      assert container_var.b.shape == {2}
      assert tensor_var.shape == {2}
    end

    test "raises a clear ArgumentError, not a KeyError, for an invalid sharding on a container leaf" do
      mesh = %Nx.Mesh{name: "mesh", shape: {2}}
      fun = fn container, tensor -> Nx.add(container.value, tensor) end

      args_list = [[%{value: Nx.tensor([1])}, Nx.tensor([1, 2])]]

      assert_raise ArgumentError, ~r/axis 1 is not valid for mesh with 1 axes/, fn ->
        Nx.Defn.shard_jit(fun, mesh,
          compiler: TestCompiler,
          # Mesh only has axis 0, so axis 1 is invalid.
          input_shardings: [%{0 => [1]}, %{}]
        ).(args_list)
      end
    end
  end

  describe "input_shardings validation" do
    test "raises when input_shardings is not a list" do
      mesh = %Nx.Mesh{name: "mesh", shape: {2}}
      fun = fn x -> Nx.add(x, 1) end

      assert_raise ArgumentError, ~r/input_shardings are required for sharding/, fn ->
        Nx.Defn.shard_jit(fun, mesh, compiler: TestCompiler, input_shardings: nil).([
          [Nx.tensor([1, 2])]
        ])
      end
    end

    test "raises when the number of input_shardings does not match the number of arguments" do
      mesh = %Nx.Mesh{name: "mesh", shape: {2}}
      fun = fn x, y -> Nx.add(x, y) end

      assert_raise ArgumentError, ~r/expected 2 input sharding configuration.*got 1/, fn ->
        Nx.Defn.shard_jit(fun, mesh, compiler: TestCompiler, input_shardings: [%{}]).([
          [Nx.tensor([1]), Nx.tensor([2])]
        ])
      end
    end
  end
end
