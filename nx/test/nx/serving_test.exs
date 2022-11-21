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

  describe "run/2" do
    test "with default callbacks" do
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
          {result, metadata}
        end)

      pre = [Nx.tensor([1, 2]), Nx.tensor([3, 4])]
      post = Nx.tensor([[2, 4], [6, 8]])
      assert Nx.Serving.run(serving, pre) == {post, :metadata}

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
      serving = Nx.Serving.new(Simple, self())
      start_supervised!({Nx.Serving, [name: config.test, serving: serving] ++ opts})
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

    test "2+2+pad=8", config do
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
      assert batch.pad == 4
    end

    test "3+4+5=6+6", config do
      simple_supervised!(config, batch_size: 6, batch_timeout: 100)

      t1 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([11, 12, 13])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([22, 24, 26])
        end)

      t2 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([21, 22, 23, 24])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([42, 44, 46, 48])
        end)


      t3 =
        Task.async(fn ->
          batch = Nx.Batch.concatenate([Nx.tensor([31, 32, 33, 34, 35])])
          assert Nx.Serving.batched_run(config.test, batch) == Nx.tensor([62, 64, 66, 68, 70])
        end)

      Task.await(t1, :infinity)
      Task.await(t2, :infinity)
      Task.await(t3, :infinity)

      assert_received {:init, :process}
      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 6

      assert_received {:batch, batch}
      assert_received :execute
      assert batch.size == 6
    end
  end
end
