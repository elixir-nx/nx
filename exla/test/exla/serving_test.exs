defmodule EXLA.ServingTest do
  use EXLA.Case, async: true

  defmodule ExecuteSync do
    @behaviour Nx.Serving

    @impl true
    def init(_type, pid, partitions) do
      funs = Enum.map(partitions, fn opts -> Nx.Defn.jit(&Nx.multiply(&1, 2), opts) end)
      {:ok, {funs, pid}}
    end

    @impl true
    def handle_batch(batch, partition, {funs, pid}) do
      jit = Enum.fetch!(funs, partition)

      fun = fn ->
        send(pid, {:execute, partition, self()})

        receive do
          :crash -> raise "oops"
          :continue -> {jit.(batch), :metadata}
        end
      end

      {:execute, fun, {funs, pid}}
    end
  end

  defp execute_sync_supervised!(config, opts) do
    serving = Nx.Serving.new(ExecuteSync, self())
    opts = [name: config.test, serving: serving, shutdown: 1000] ++ opts
    start_supervised!({Nx.Serving, opts})
  end

  describe "batched_run" do
    test "2+2=4", config do
      execute_sync_supervised!(config, batch_size: 2)

      task1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([1, 2])])
          tensor = Nx.Serving.batched_run(config.test, batch)
          assert is_struct(tensor.data, EXLA.Backend)
          assert_equal(tensor, Nx.tensor([2, 4]))
          tensor.data.buffer.device_id
        end)

      task2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([3, 4])])
          tensor = Nx.Serving.batched_run(config.test, batch)
          assert is_struct(tensor.data, EXLA.Backend)
          assert_equal(tensor, Nx.tensor([6, 8]))
          tensor.data.buffer.device_id
        end)

      assert_receive {:execute, 0, executor}, 5000
      send(executor, :continue)

      assert_receive {:execute, 0, executor}, 5000
      send(executor, :continue)

      assert Task.await(task1) == Task.await(task2)
    end
  end

  describe "partitioning" do
    @describetag :multi_device
    @describetag :mlir_multi_device_error
    for backend <- [Nx.BinaryBackend, EXLA.Backend] do
      test "spawns tasks concurrently with #{inspect(backend)}", config do
        execute_sync_supervised!(config, batch_size: 2, partitions: true)

        task1 =
          Task.async(fn ->
            batch = Nx.Batch.concatenate([Nx.tensor([1, 2], backend: unquote(backend))])
            tensor = Nx.Serving.batched_run(config.test, batch)
            assert is_struct(tensor.data, EXLA.Backend)
            assert_equal(tensor, Nx.tensor([2, 4]))
            tensor.data.buffer.device_id
          end)

        task2 =
          Task.async(fn ->
            batch = Nx.Batch.concatenate([Nx.tensor([3, 4], backend: unquote(backend))])
            tensor = Nx.Serving.batched_run(config.test, batch)
            assert is_struct(tensor.data, EXLA.Backend)
            assert_equal(tensor, Nx.tensor([6, 8]))
            tensor.data.buffer.device_id
          end)

        assert_receive {:execute, 0, executor1}, 5000
        assert_receive {:execute, 1, executor2}, 5000
        send(executor1, :continue)
        send(executor2, :continue)

        assert Task.await(task1) != Task.await(task2)
      end
    end
  end
end
