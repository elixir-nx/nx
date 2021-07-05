defmodule Nx.Defn.StreamTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  defn defn_sum(entry, acc), do: {acc, entry + acc}

  def elixir_sum(entry, acc) do
    true = Process.get(Nx.Defn.Compiler) in [Evaluator, Identity]
    {acc, entry + acc}
  end

  test "runs defn stream" do
    %_{} = stream = Nx.Defn.stream(&defn_sum/2, [0, 0])
    assert Nx.Stream.send(stream, 1) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(0)

    assert Nx.Stream.send(stream, 2) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(1)

    assert Nx.Stream.done(stream) == Nx.tensor(3)
  end

  # test "runs defn function async" do
  #   assert %_{} = async = Nx.Defn.async(&defn_async/2, [{4, 5}, 3])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)

  #   assert %_{} = async = Nx.Defn.async(&defn_async/2, [{4, 5}, Nx.tensor(3)])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)
  #   assert %_{} = async = Nx.Defn.async(&defn_async(&1, 3), [{4, 5}])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)
  # end

  # test "runs elixir function async" do
  #   assert %_{} = async = Nx.Defn.async(&elixir_async/2, [{4, 5}, 3])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)

  #   assert %_{} = async = Nx.Defn.async(&elixir_async/2, [{4, 5}, Nx.tensor(3)])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)

  #   assert %_{} = async = Nx.Defn.async(&elixir_async(&1, 3), [{4, 5}])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)
  # end

  # @tag :capture_log
  # test "raises on errors" do
  #   Process.flag(:trap_exit, true)
  #   assert %_{} = async = Nx.Defn.async(fn -> :ok end, [])

  #   ref = Process.monitor(async.pid)
  #   assert_receive {:DOWN, ^ref, _, _, _}

  #   assert catch_exit(Nx.Async.await!(async)) == {:noproc, {Nx.Async, :await!, [async]}}
  #   assert_receive {:EXIT, _, _}
  # end

  # test "raises if already awaited" do
  #   assert %_{} = async = Nx.Defn.async(&defn_async/2, [{4, 5}, 3])
  #   assert Nx.Async.await!(async) == Nx.tensor(6)
  #   assert catch_exit(Nx.Async.await!(async)) == {:noproc, {Nx.Async, :await!, [async]}}
  # end

  # defn async_iota(), do: Nx.iota({3, 3})

  # @tag :capture_log
  # test "uses the default backend on iota" do
  #   Process.flag(:trap_exit, true)
  #   Nx.default_backend(UnknownBackend)
  #   assert %_{} = Nx.Defn.async(&async_iota/0, [])
  #   assert_receive {:EXIT, _, {:undef, _}}
  # end
end
