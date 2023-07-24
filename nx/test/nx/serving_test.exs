defmodule Nx.ServingTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  defmodule Simple do
    @behaviour Nx.Serving

    @impl true
    def init(type, pid, partitions) do
      send(pid, {:init, type, partitions})
      {:ok, pid}
    end

    @impl true
    def handle_batch(batch, partition, pid) do
      send(pid, {:batch, partition, batch})

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
    def init(_type, pid, _partitions) do
      {:ok, pid}
    end

    @impl true
    def handle_batch(batch, partition, pid) do
      fun = fn ->
        send(pid, {:execute, partition, self()})

        receive do
          :crash -> raise "oops"
          :continue -> {Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [batch]), :metadata}
        end
      end

      {:execute, fun, pid}
    end
  end

  defn add_five_round_about(batch) do
    batch
    |> Nx.multiply(2)
    |> hook(:double)
    |> Nx.add(10)
    |> hook(:plus_ten)
    |> Nx.divide(2)
    |> hook(:to_be_ignored)
  end

  describe "run/2" do
    test "with function" do
      serving = Nx.Serving.new(fn opts -> Nx.Defn.jit(&Nx.multiply(&1, 2), opts) end)
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
    end

    test "with function and batch key" do
      serving =
        Nx.Serving.new(fn batch_key, opts ->
          send(self(), {:batch_key, batch_key})
          Nx.Defn.jit(&Nx.multiply(&1, 2), opts)
        end)

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])]) |> Nx.Batch.key(:foo)
      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
      assert_receive {:batch_key, :foo}

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])]) |> Nx.Batch.key(:bar)
      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
      assert_receive {:batch_key, :bar}
    end

    test "with container (and jit)" do
      serving = Nx.Serving.jit(fn {a, b} -> {Nx.multiply(a, 2), Nx.divide(b, 2)} end)
      batch = Nx.Batch.concatenate([{Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])}])
      assert Nx.Serving.run(serving, batch) == {Nx.tensor([2, 4, 6]), Nx.tensor([2, 2.5, 3])}
    end

    test "with padding" do
      serving =
        Nx.Serving.new(fn opts ->
          fn batch ->
            Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [Nx.Batch.pad(batch, 4)], opts)
          end
        end)

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6], [8, 10, 12]])
    end

    test "with module callbacks" do
      serving = Nx.Serving.new(Simple, self(), garbage_collect: 1)
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

      assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
      assert_received {:init, :inline, [[batch_keys: [:default], garbage_collect: 1]]}
      assert_received {:batch, 0, batch}
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
        |> Nx.Serving.client_postprocessing(fn {result, metadata}, info ->
          send(self(), {:post, result, metadata, info})
          {result, metadata, info}
        end)

      pre = [Nx.tensor([1, 2]), Nx.tensor([3, 4])]
      post = Nx.tensor([[2, 4], [6, 8]])
      assert Nx.Serving.run(serving, pre) == {post, :metadata, :preprocessing!}

      assert_received {:init, :inline, [[batch_keys: [:default]]]}
      assert_received {:pre, ^pre}
      assert_received {:batch, 0, batch}
      assert_received :execute
      assert_received {:post, ^post, :metadata, :preprocessing!}
      assert batch.size == 2
      assert batch.pad == 0
      assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) == Nx.tensor([[1, 2], [3, 4]])
    end

    test "instrumenting with telemetry" do
      ref =
        :telemetry_test.attach_event_handlers(
          self(),
          [
            [:nx, :serving, :execute, :stop],
            [:nx, :serving, :preprocessing, :stop],
            [:nx, :serving, :postprocessing, :stop]
          ]
        )

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

      fn opts -> Nx.Defn.jit(&Nx.multiply(&1, 2), opts) end
      |> Nx.Serving.new()
      |> Nx.Serving.client_preprocessing(fn _entry -> {batch, :pre} end)
      |> Nx.Serving.client_postprocessing(fn {res, meta}, info -> {res, meta, info} end)
      |> Nx.Serving.run(batch)

      assert_receive {[:nx, :serving, :execute, :stop], ^ref, _measure, meta}
      assert %{metadata: :server_info, module: Nx.Serving.Default} = meta

      assert_receive {[:nx, :serving, :preprocessing, :stop], ^ref, _measure, meta}
      assert %{info: :pre, input: %Nx.Batch{}} = meta

      assert_receive {[:nx, :serving, :postprocessing, :stop], ^ref, _measure, meta}
      assert %{info: :pre} = meta
    end
  end

  describe "run + streaming" do
    test "with function" do
      serving =
        Nx.Serving.new(fn opts -> Nx.Defn.jit(&Nx.multiply(&1, 2), opts) end)
        |> Nx.Serving.streaming()

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

      assert Nx.Serving.run(serving, batch) |> Enum.to_list() ==
               [{:done, Nx.tensor([[2, 4, 6]]), :server_info}]
    end

    test "with padding" do
      serving =
        Nx.Serving.new(fn opts ->
          fn batch ->
            Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [Nx.Batch.pad(batch, 4)], opts)
          end
        end)
        |> Nx.Serving.streaming()

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])

      assert Nx.Serving.run(serving, batch) |> Enum.to_list() ==
               [{:done, Nx.tensor([[2, 4, 6], [8, 10, 12]]), :server_info}]
    end

    test "with module callbacks" do
      serving = Nx.Serving.new(Simple, self(), garbage_collect: 1) |> Nx.Serving.streaming()
      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

      assert Nx.Serving.run(serving, batch) |> Enum.to_list() ==
               [{:done, Nx.tensor([[2, 4, 6]]), :metadata}]

      assert_received {:init, :inline, [[batch_keys: [:default], hooks: %{}, garbage_collect: 1]]}
      assert_received {:batch, 0, batch}
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
        |> Nx.Serving.client_postprocessing(fn stream, info ->
          send(self(), {:post, stream, info})
          {stream, info}
        end)
        |> Nx.Serving.streaming()

      pre = [Nx.tensor([1, 2]), Nx.tensor([3, 4])]
      {stream, :preprocessing!} = Nx.Serving.run(serving, pre)
      assert Enum.to_list(stream) == [{:done, Nx.tensor([[2, 4], [6, 8]]), :metadata}]

      assert_received {:init, :inline, [[batch_keys: [:default], hooks: %{}]]}
      assert_received {:pre, ^pre}
      assert_received {:batch, 0, batch}
      assert_received :execute
      assert_received {:post, ^stream, :preprocessing!}
      assert batch.size == 2
      assert batch.pad == 0
      assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) == Nx.tensor([[1, 2], [3, 4]])
    end

    test "with hooks" do
      serving =
        Nx.Serving.jit(&add_five_round_about/1)
        |> Nx.Serving.streaming(hooks: [:double, :plus_ten])

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])

      assert Nx.Serving.run(serving, batch) |> Enum.to_list() == [
               {:double, Nx.tensor([[2, 4, 6], [8, 10, 12]])},
               {:plus_ten, Nx.tensor([[12, 14, 16], [18, 20, 22]])},
               {:done, Nx.tensor([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]), :server_info}
             ]
    end
  end

  defp simple_supervised!(config, opts \\ []) do
    opts = Keyword.put_new(opts, :serving, Nx.Serving.new(Simple, self()))
    start_supervised!({Nx.Serving, [name: config.test] ++ opts})
  end

  describe "batched_run" do
    test "supervision tree", config do
      pid = simple_supervised!(config)
      :sys.get_status(config.test)
      [_, _] = Supervisor.which_children(pid)
    end

    test "1=1", config do
      simple_supervised!(config, serving: Nx.Serving.new(Simple, self(), garbage_collect: true))

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[2, 4, 6]])
      assert_received {:init, :process, [[batch_keys: [:default], garbage_collect: true]]}
      assert_received {:batch, 0, batch}
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

      assert_received {:init, :process, [[batch_keys: [:default]]]}
      assert_received {:batch, 0, batch}
      assert_received :execute
      assert batch.size == 4
      assert batch.pad == 0
    end

    test "2+2=4 with pre and post", config do
      serving =
        Nx.Serving.new(Simple, self())
        |> Nx.Serving.process_options(batch_size: 4)
        |> Nx.Serving.client_preprocessing(fn entry ->
          {Nx.Batch.stack(entry), :preprocessing!}
        end)
        |> Nx.Serving.client_postprocessing(fn {result, metadata}, info ->
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

      assert_received {:init, :process, [[batch_keys: [:default]]]}
      assert_received {:batch, 0, batch}
      assert_received :execute
      assert batch.size == 4
      assert batch.pad == 0
    end

    test "3+4+5=6+6 (container)", config do
      serving =
        Nx.Serving.new(fn opts ->
          Nx.Defn.jit(fn {a, b} -> {Nx.multiply(a, 2), Nx.divide(b, 2)} end, opts)
        end)
        |> Nx.Serving.process_options(batch_size: 6)

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
      opts = [name: config.test, serving: serving, shutdown: 1000] ++ opts
      start_supervised!({Nx.Serving, opts})
    end

    @tag :capture_log
    test "1=>crash", config do
      execute_sync_supervised!(config)

      {_pid, ref} =
        spawn_monitor(fn ->
          batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
          Nx.Serving.batched_run(config.test, batch)
        end)

      assert_receive {:execute, 0, executor}
      send(executor, :crash)

      assert_receive {:DOWN, ^ref, _, _,
                      {{%RuntimeError{}, _}, {Nx.Serving, :local_batched_run, [_, _]}}}
    end

    @tag :capture_log
    test "2+3=>crash", config do
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

      assert_receive {:execute, 0, executor}
      send(executor, :continue)

      assert_receive {:execute, 0, executor}
      send(executor, :crash)

      # One task should succeed and the other terminate
      assert_receive {:DOWN, ref, _, _,
                      {{%RuntimeError{}, _}, {Nx.Serving, :local_batched_run, [_, _]}}}
                     when ref in [ref1, ref2]

      assert_receive {:DOWN, ref, _, _, :normal} when ref in [ref1, ref2]
      refute_received {:execute, _partition, _executor}
    end

    @tag :capture_log
    test "2=>shutdown=>2 (queued)", config do
      serving_pid = execute_sync_supervised!(config, batch_timeout: 100, batch_size: 2)

      {_pid, ref1} =
        spawn_monitor(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([2, 4])
        end)

      {_pid, ref2} =
        spawn_monitor(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([6, 8])
        end)

      assert_receive {:execute, 0, executor}
      send(serving_pid, {:system, {self(), make_ref()}, {:terminate, :shutdown}})
      send(executor, :continue)

      # One task should succeed and the other terminate
      assert_receive {:DOWN, ref, _, _, :normal}
                     when ref in [ref1, ref2]

      assert_receive {:DOWN, ref, _, _, {:noproc, {Nx.Serving, :local_batched_run, [_, _]}}}
                     when ref in [ref1, ref2]

      refute_received {:execute, _partition, _executor}
    end

    @tag :capture_log
    test "2=>shutdown=>1 (stacked)", config do
      serving_pid = execute_sync_supervised!(config, batch_timeout: 100, batch_size: 2)

      {_pid, ref1} =
        spawn_monitor(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([2, 4])
        end)

      {_pid, ref2} =
        spawn_monitor(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([6])
        end)

      assert_receive {:execute, 0, executor}
      send(serving_pid, {:system, {self(), make_ref()}, {:terminate, :shutdown}})
      send(executor, :continue)

      # One task should succeed and the other terminate
      assert_receive {:DOWN, ref, _, _, :normal}
                     when ref in [ref1, ref2]

      assert_receive {:DOWN, ref, _, _, {:noproc, {Nx.Serving, :local_batched_run, [_, _]}}}
                     when ref in [ref1, ref2]

      refute_received {:execute, _partition, _executor}
    end

    test "2=>2=>1+timeout", config do
      execute_sync_supervised!(config, batch_timeout: 100, batch_size: 2)

      task1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([2, 4])
        end)

      task2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([6, 8])
        end)

      task3 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([5])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([10])
        end)

      assert_receive {:execute, 0, executor1}
      send(executor1, :continue)

      assert_receive {:execute, 0, executor2}
      send(executor2, :continue)

      assert_receive {:execute, 0, executor3}
      send(executor3, :continue)

      Task.await(task1)
      Task.await(task2)
      Task.await(task3)
    end

    test "batch keys", config do
      serving =
        Nx.Serving.new(fn
          :double, opts -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({3}, :s64)], opts)
          :half, opts -> Nx.Defn.compile(&Nx.divide(&1, 2), [Nx.template({3}, :s64)], opts)
        end)

      simple_supervised!(config,
        serving: serving,
        batch_timeout: 10_000,
        batch_size: 3,
        batch_keys: [:double, :half]
      )

      assert_raise ArgumentError,
                   "unknown batch key: :default (expected one of [:double, :half])",
                   fn ->
                     Nx.Serving.batched_run(config.test, Nx.Batch.concatenate([Nx.iota({3})]))
                   end

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.iota({2})]) |> Nx.Batch.key(:double)
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([0, 2])
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.iota({3})]) |> Nx.Batch.key(:half)
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([0.0, 0.5, 1.0])
        end)

      Task.await(t2, :infinity)
      refute Task.yield(t1, 0)

      t3 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.iota({1})]) |> Nx.Batch.key(:double)
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([0])
        end)

      Task.await(t1, :infinity)
      Task.await(t3, :infinity)
    end

    test "with padding", config do
      serving =
        Nx.Serving.new(fn opts ->
          fn batch ->
            Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [Nx.Batch.pad(batch, 4)], opts)
          end
        end)

      simple_supervised!(config, serving: serving, batch_size: 4, batch_timeout: 100)

      # Partial batch
      t1 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([11, 12])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[22, 24]])
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([21, 22])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([[42, 44]])
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)

      batch = Nx.Batch.concatenate([Nx.tensor([[11, 12], [13, 14], [15, 16], [17, 18]])])

      assert Nx.Serving.batched_run(config.test, batch) ==
               Nx.tensor([[22, 24], [26, 28], [30, 32], [34, 36]])
    end

    test "instrumenting with telemetry", config do
      ref = :telemetry_test.attach_event_handlers(self(), [[:nx, :serving, :execute, :stop]])

      simple_supervised!(config)

      batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

      Nx.Serving.batched_run(config.test, batch)

      assert_receive {[:nx, :serving, :execute, :stop], ^ref, _measure, meta}
      assert %{metadata: :metadata, module: Nx.ServingTest.Simple} = meta
    end

    test "conflict on batch size" do
      assert_raise ArgumentError,
                   ~r":batch_size has been set when starting an Nx.Serving process \(15\) but a conflicting value was already set on the Nx.Serving struct \(30\)",
                   fn ->
                     serving =
                       Nx.Serving.new(Simple, self())
                       |> Nx.Serving.process_options(batch_size: 30)

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

  describe "batched_run + streaming" do
    test "2+2=4 with pre and post", config do
      serving =
        Nx.Serving.new(Simple, self())
        |> Nx.Serving.process_options(batch_size: 4)
        |> Nx.Serving.client_preprocessing(fn entry ->
          {Nx.Batch.stack(entry), :preprocessing!}
        end)
        |> Nx.Serving.client_postprocessing(fn stream, info ->
          {stream, info}
        end)
        |> Nx.Serving.streaming()

      simple_supervised!(config, serving: serving, batch_timeout: 10_000)

      t1 =
        Task.async(fn ->
          batch = [Nx.tensor([11, 12]), Nx.tensor([13, 14])]

          {stream, :preprocessing!} = Nx.Serving.batched_run(config.test, batch)
          assert Enum.to_list(stream) == [{:done, Nx.tensor([[22, 24], [26, 28]]), :metadata}]
        end)

      t2 =
        Task.async(fn ->
          batch = [Nx.tensor([21, 22]), Nx.tensor([23, 24])]

          {stream, :preprocessing!} = Nx.Serving.batched_run(config.test, batch)
          assert Enum.to_list(stream) == [{:done, Nx.tensor([[42, 44], [46, 48]]), :metadata}]
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)

      assert_received {:init, :process, [[batch_keys: [:default], hooks: %{}]]}
      assert_received {:batch, 0, batch}
      assert_received :execute
      assert batch.size == 4
      assert batch.pad == 0
    end

    @tag :capture_log
    test "1=>crash", config do
      serving = Nx.Serving.new(ExecuteSync, self()) |> Nx.Serving.streaming()
      simple_supervised!(config, serving: serving, shutdown: 1000)

      {_pid, ref} =
        spawn_monitor(fn ->
          batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
          Enum.to_list(Nx.Serving.batched_run(config.test, batch))
        end)

      assert_receive {:execute, 0, executor}
      send(executor, :crash)

      assert_receive {:DOWN, ^ref, _, _, {{%RuntimeError{}, _}, {Nx.Serving, :streaming, []}}}
    end

    test "2+2=2(+pad)+2(+pad) and hooks", config do
      serving =
        Nx.Serving.jit(&add_five_round_about/1)
        |> Nx.Serving.streaming(hooks: [:double, :plus_ten])

      simple_supervised!(config, serving: serving, batch_timeout: 100, batch_size: 3)

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([11, 12]), Nx.tensor([13, 14])])

          assert Nx.Serving.batched_run(config.test, batch) |> Enum.to_list() == [
                   {:double, Nx.tensor([[22, 24], [26, 28]])},
                   {:plus_ten, Nx.tensor([[32, 34], [36, 38]])},
                   {:done, Nx.tensor([[16.0, 17.0], [18.0, 19.0]]), :server_info}
                 ]
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([21, 22]), Nx.tensor([23, 24])])

          assert Nx.Serving.batched_run(config.test, batch) |> Enum.to_list() == [
                   {:double, Nx.tensor([[42, 44], [46, 48]])},
                   {:plus_ten, Nx.tensor([[52, 54], [56, 58]])},
                   {:done, Nx.tensor([[26.0, 27.0], [28.0, 29.0]]), :server_info}
                 ]
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)
    end

    test "2+2=4(+pad) and hooks", config do
      serving =
        Nx.Serving.jit(&add_five_round_about/1)
        |> Nx.Serving.streaming(hooks: [:double, :plus_ten])

      simple_supervised!(config, serving: serving, batch_timeout: 100, batch_size: 5)

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([11, 12]), Nx.tensor([13, 14])])

          assert Nx.Serving.batched_run(config.test, batch) |> Enum.to_list() == [
                   {:double, Nx.tensor([[22, 24], [26, 28]])},
                   {:plus_ten, Nx.tensor([[32, 34], [36, 38]])},
                   {:done, Nx.tensor([[16.0, 17.0], [18.0, 19.0]]), :server_info}
                 ]
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.stack([Nx.tensor([21, 22]), Nx.tensor([23, 24])])

          assert Nx.Serving.batched_run(config.test, batch) |> Enum.to_list() == [
                   {:double, Nx.tensor([[42, 44], [46, 48]])},
                   {:plus_ten, Nx.tensor([[52, 54], [56, 58]])},
                   {:done, Nx.tensor([[26.0, 27.0], [28.0, 29.0]]), :server_info}
                 ]
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)
    end
  end

  describe "partitioning" do
    test "spawns tasks concurrently", config do
      serving = Nx.Serving.new(ExecuteSync, self(), max_concurrency: 2)

      opts = [
        name: config.test,
        serving: serving,
        partitions: true,
        batch_size: 2,
        shutdown: 1000
      ]

      start_supervised!({Nx.Serving, opts})

      task1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([2, 4])
        end)

      task2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([6, 8])
        end)

      assert_receive {:execute, 0, executor1}
      assert_receive {:execute, 1, executor2}
      send(executor1, :continue)
      send(executor2, :continue)

      assert Task.await(task1)
      assert Task.await(task2)
    end
  end

  describe "distributed" do
    setup do
      :pg.monitor_scope(Nx.Serving.PG)
      :ok
    end

    test "spawns distributed tasks locally", config do
      parent = self()

      preprocessing = fn input ->
        send(parent, {:pre, input})
        input
      end

      postprocessing = fn output ->
        send(parent, {:post, output})
        output
      end

      serving =
        Nx.Serving.new(ExecuteSync, self())
        |> Nx.Serving.distributed_postprocessing(postprocessing)

      opts = [
        name: config.test,
        serving: serving,
        batch_size: 2,
        shutdown: 1000
      ]

      start_supervised!({Nx.Serving, opts})

      task1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])

          assert Nx.Serving.batched_run({:distributed, config.test}, batch, preprocessing) ==
                   Nx.tensor([2, 4])
        end)

      task2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4])])

          assert Nx.Serving.batched_run({:distributed, config.test}, batch, preprocessing) ==
                   Nx.tensor([6, 8])
        end)

      assert_receive {:pre, %Nx.Batch{size: 2}}
      assert_receive {:execute, 0, executor}
      send(executor, :continue)
      assert_receive {:post, %Nx.Tensor{}}

      assert_receive {:pre, %Nx.Batch{size: 2}}
      assert_receive {:execute, 0, executor}
      send(executor, :continue)
      assert_receive {:post, %Nx.Tensor{}}

      assert Task.await(task1)
      assert Task.await(task2)
    end

    @tag :distributed
    test "spawns distributed tasks over the network", config do
      parent = self()

      preprocessing = fn input ->
        send(parent, {:pre, node(), input})
        input
      end

      opts = [
        name: config.test,
        batch_size: 2,
        shutdown: 1000
      ]

      Node.spawn_link(:"secondary@127.0.0.1", DistributedServings, :multiply, [parent, opts])
      assert_receive {_, :join, Nx.Serving, _}

      batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])

      # local call
      assert {:noproc, _} =
               catch_exit(Nx.Serving.batched_run({:local, config.test}, batch, preprocessing))

      # distributed call
      assert Nx.Serving.batched_run({:distributed, config.test}, batch, preprocessing) ==
               Nx.tensor([2, 4])

      assert_receive {:pre, node, %Nx.Batch{size: 2}} when node == node()
      assert_receive {:post, node, tensor} when node != node()
      assert tensor == Nx.tensor([2, 4])

      # lookup call
      batch = Nx.Batch.concatenate([Nx.tensor([3])])
      assert Nx.Serving.batched_run(config.test, batch, preprocessing) == Nx.tensor([6])
      assert_receive {:pre, node, %Nx.Batch{size: 1}} when node == node()
      assert_receive {:post, node, tensor} when node != node()
      assert tensor == Nx.tensor([6])
    end

    @tag :distributed
    test "spawns distributed tasks over the network with streaming", config do
      parent = self()

      preprocessing = fn input ->
        send(parent, {:pre, node(), input})
        input
      end

      opts = [
        name: config.test,
        batch_size: 2,
        shutdown: 1000
      ]

      args = [parent, opts]
      Node.spawn_link(:"secondary@127.0.0.1", DistributedServings, :add_five_round_about, args)
      assert_receive {_, :join, Nx.Serving, _}

      batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])

      # local call
      assert {:noproc, _} =
               catch_exit(Nx.Serving.batched_run({:local, config.test}, batch, preprocessing))

      # distributed call
      assert Nx.Serving.batched_run({:distributed, config.test}, batch, preprocessing)
             |> Enum.to_list() == [
               {{:double, Nx.tensor([2, 4])}, :"secondary@127.0.0.1"},
               {{:plus_ten, Nx.tensor([12, 14])}, :"secondary@127.0.0.1"},
               {{:done, Nx.tensor([6.0, 7.0]), :server_info}, :"secondary@127.0.0.1"}
             ]

      assert_receive {:pre, node, %Nx.Batch{size: 2}} when node == node()

      # lookup call
      batch = Nx.Batch.concatenate([Nx.tensor([3])])

      assert Nx.Serving.batched_run(config.test, batch, preprocessing) |> Enum.to_list() == [
               {{:double, Nx.tensor([6])}, :"secondary@127.0.0.1"},
               {{:plus_ten, Nx.tensor([16])}, :"secondary@127.0.0.1"},
               {{:done, Nx.tensor([8.0]), :server_info}, :"secondary@127.0.0.1"}
             ]

      assert_receive {:pre, node, %Nx.Batch{size: 1}} when node == node()
    end

    @tag :distributed
    test "spawns distributed tasks over the network with hidden nodes", config do
      parent = self()

      preprocessing = fn input ->
        send(parent, {:pre, node(), input})
        input
      end

      opts = [
        name: config.test,
        batch_size: 2,
        shutdown: 1000
      ]

      Node.spawn_link(:"tertiary@127.0.0.1", DistributedServings, :multiply, [parent, opts])
      assert_receive {_, :join, Nx.Serving, _}

      batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])

      # Make sure stray messages do not crash processing
      send(self(), {:DOWN, make_ref(), :process, self(), :normal})

      assert Nx.Serving.batched_run({:distributed, config.test}, batch, preprocessing) ==
               Nx.tensor([2, 4])

      assert_receive {:pre, node, %Nx.Batch{size: 2}} when node == node()
      assert_receive {:post, node, tensor} when node != node()
      assert tensor == Nx.tensor([2, 4])
    end
  end
end
