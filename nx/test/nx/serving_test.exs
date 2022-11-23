defmodule Nx.ServingTest do
  use ExUnit.Case, async: true

  defmodule Simple do
    @behaviour Nx.Serving

    @impl true
    def init(type, pid) do
      send(pid, {:init, type})
      {:ok, pid}
    end

    @impl true
    def handle_batch(batch, pid) do
      send(pid, {:batch, batch})

      fun = fn ->
        send(pid, :execute)
        {Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [batch]), :metadata}
      end

      {:execute, fun, pid}
    end
  end

  defmodule ExecuteSync do
    @behaviour Nx.Serving

    @impl true
    def init(_type, pid) do
      {:ok, pid}
    end

    @impl true
    def handle_batch(batch, pid) do
      send(pid, :batch)

      fun = fn ->
        send(pid, {:execute, self()})

        receive do
          :crash -> raise "oops"
          :continue -> {Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [batch]), :metadata}
        end
      end

      {:execute, fun, pid}
    end
  end

  describe "run/2" do
    test "with function" do
      serving = Nx.Serving.new(fn -> Nx.Defn.jit(&Nx.multiply(&1, 2)) end)
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
    end

    test "with container" do
      serving =
        Nx.Serving.new(fn ->
          Nx.Defn.jit(fn {a, b} -> {Nx.multiply(a, 2), Nx.divide(b, 2)} end)
        end)

      batch = Nx.Batch.concatenate([{Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])}])
      assert Nx.Serving.run(serving, batch) == {Nx.tensor([2, 4, 6]), Nx.tensor([2, 2.5, 3])}
    end

    test "with module callbacks" do
      serving = Nx.Serving.new(Simple, self())
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
      assert_received {:init, :inline}
      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 1
      assert batch.pad == 0
      assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) == Nx.tensor([[1, 2, 3]])
    end

    test "with client pre/post" do
      serving =
        Nx.Serving.new(Simple, self())
        |> Nx.Serving.client_preprocessing(fn entry ->
          send(self(), {:pre, entry})
          {Nx.Batch.stack(entry), :preprocessing!}
        end)
        |> Nx.Serving.client_postprocessing(fn result, metadata, info ->
          send(self(), {:post, result, metadata, info})
          {result, metadata, info}
        end)

      pre = [Nx.tensor([1, 2]), Nx.tensor([3, 4])]
      post = Nx.tensor([[2, 4], [6, 8]])
      assert Nx.Serving.run(serving, pre) == {post, :metadata, :preprocessing!}

      assert_received {:init, :inline}
      assert_received {:pre, ^pre}
      assert_received {:batch, batch}
      assert_received :execute
      assert_received {:post, ^post, :metadata, :preprocessing!}
      assert batch.size == 2
      assert batch.pad == 0
      assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) == Nx.tensor([[1, 2], [3, 4]])
    end
  end

  describe "batched_run" do
    defp simple_supervised!(config, opts \\ []) do
      opts = Keyword.put_new(opts, :serving, Nx.Serving.new(Simple, self()))
      start_supervised!({Nx.Serving, [name: config.test] ++ opts})
    end

    test "supervision tree", config do
      pid = simple_supervised!(config)
      [_, _] = Supervisor.which_children(pid)
    end

    test "1=1", config do
      simple_supervised!(config)

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[2, 4, 6]])
      assert_received {:init, :process}
      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 1
      assert batch.pad == 0
      assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) == Nx.tensor([[1, 2, 3]])
    end

    test "2+2=4", config do
      simple_supervised!(config, batch_size: 4, batch_timeout: 10_000)

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([11, 12]), Nx.tensor([13, 14])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[22, 24], [26, 28]])
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([21, 22]), Nx.tensor([23, 24])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[42, 44], [46, 48]])
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)

      assert_received {:init, :process}
      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 4
      assert batch.pad == 0
    end

    test "2+2=4 with pre and post", config do
      serving =
        Nx.Serving.new(Simple, self(), batch_size: 4)
        |> Nx.Serving.client_preprocessing(fn entry ->
          {Nx.Batch.stack(entry), :preprocessing!}
        end)
        |> Nx.Serving.client_postprocessing(fn result, metadata, info ->
          {result, metadata, info}
        end)

      simple_supervised!(config, serving: serving, batch_timeout: 10_000)

      t1 =
        Task.async(fn ->
          batch = [Nx.tensor([11, 12]), Nx.tensor([13, 14])]

          assert Nx.Serving.batched_run(config.test, batch) ==
                   {Nx.tensor([[22, 24], [26, 28]]), :metadata, :preprocessing!}
        end)

      t2 =
        Task.async(fn ->
          batch = [Nx.tensor([21, 22]), Nx.tensor([23, 24])]

          assert Nx.Serving.batched_run(config.test, batch) ==
                   {Nx.tensor([[42, 44], [46, 48]]), :metadata, :preprocessing!}
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)

      assert_received {:init, :process}
      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 4
      assert batch.pad == 0
    end

    test "2+2+timeout=8", config do
      simple_supervised!(config, batch_size: 8, batch_timeout: 100)

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([11, 12]), Nx.tensor([13, 14])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[22, 24], [26, 28]])
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([21, 22]), Nx.tensor([23, 24])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[42, 44], [46, 48]])
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)

      assert_received {:init, :process}
      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 4
      assert batch.pad == 0
    end

    test "3+4+5=6+6 (container)", config do
      serving =
        Nx.Serving.new(
          fn -> Nx.Defn.jit(fn {a, b} -> {Nx.multiply(a, 2), Nx.divide(b, 2)} end) end,
          batch_size: 6
        )

      simple_supervised!(config, batch_timeout: 100, serving: serving)

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([{Nx.tensor([11, 12, 13]), Nx.tensor([14, 15, 16])}])

          assert Nx.Serving.batched_run(config.test, batch) ==
                   {Nx.tensor([22, 24, 26]), Nx.tensor([7, 7.5, 8])}
        end)

      t2 =
        Task.async(fn ->
          batch =
            Nx.Batch.concatenate([{Nx.tensor([21, 22, 23, 24]), Nx.tensor([25, 26, 27, 28])}])

          assert Nx.Serving.batched_run(config.test, batch) ==
                   {Nx.tensor([42, 44, 46, 48]), Nx.tensor([12.5, 13, 13.5, 14])}
        end)

      t3 =
        Task.async(fn ->
          batch =
            Nx.Batch.concatenate([
              {Nx.tensor([31, 32, 33, 34, 35]), Nx.tensor([36, 37, 38, 39, 40])}
            ])

          assert Nx.Serving.batched_run(config.test, batch) ==
                   {Nx.tensor([62, 64, 66, 68, 70]), Nx.tensor([18, 18.5, 19, 19.5, 20])}
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)
      Task.await(t3, :infinity)
    end

    defp execute_sync_supervised!(config, opts \\ []) do
      serving = Nx.Serving.new(ExecuteSync, self())
      start_supervised!({Nx.Serving, [name: config.test, serving: serving] ++ opts})
    end

    @tag :capture_log
    test "1=crash", config do
      execute_sync_supervised!(config)

      {_pid, ref} =
        spawn_monitor(fn ->
          batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
          Nx.Serving.batched_run(config.test, batch)
        end)

      assert_receive {:execute, executor}
      send(executor, :crash)

      assert_receive {:DOWN, ^ref, _, _,
                      {{%RuntimeError{}, _}, {Nx.Serving, :batched_run, [_, _]}}}
    end

    @tag :capture_log
    test "2+3=crash", config do
      execute_sync_supervised!(config, batch_timeout: 100, batch_size: 4)

      {_pid, ref1} =
        spawn_monitor(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([2, 4])
        end)

      {_pid, ref2} =
        spawn_monitor(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4, 5])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([6, 8, 10])
        end)

      assert_receive {:execute, executor}
      send(executor, :continue)

      assert_receive {:execute, executor}
      send(executor, :crash)

      assert_receive {:DOWN, ref, _, _,
                      {{%RuntimeError{}, _}, {Nx.Serving, :batched_run, [_, _]}}}
                     when ref in [ref1, ref2]
    end

    test "2=>2=>1+timeout", config do
      execute_sync_supervised!(config, batch_timeout: 100, batch_size: 2)

      task1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([2, 4])
        end)

      assert_receive :batch

      task2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([6, 8])
        end)

      assert_receive :batch

      task3 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([5])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([10])
        end)

      Process.sleep(100)

      assert_receive {:execute, executor1}
      send(executor1, :continue)
      Task.await(task1)

      assert_receive {:execute, executor2}
      send(executor2, :continue)
      Task.await(task2)

      assert_receive :batch
      assert_receive {:execute, executor3}
      send(executor3, :continue)
      Task.await(task3)
    end

    test "conflict on batch size" do
      assert_raise ArgumentError,
                   ~r":batch_size has been set when starting an Nx.Serving process \(15\) but a conflicting value was already set on the Nx.Serving struct \(30\)",
                   fn ->
                     serving = Nx.Serving.new(Simple, self(), batch_size: 30)
                     Nx.Serving.start_link(name: :unused, serving: serving, batch_size: 15)
                   end
    end

    test "errors on batch size", config do
      simple_supervised!(config, batch_size: 2)

      assert_raise ArgumentError, "cannot run with empty Nx.Batch", fn ->
        Nx.Serving.batched_run(config.test, Nx.Batch.new())
      end

      assert_raise ArgumentError,
                   "batch size (3) cannot exceed Nx.Serving server batch size of 2",
                   fn ->
                     Nx.Serving.batched_run(
                       config.test,
                       Nx.Batch.concatenate([Nx.tensor([1, 2, 3])])
                     )
                   end
    end
  end
end
