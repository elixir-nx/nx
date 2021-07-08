defmodule Nx.Defn.StreamTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  defn defn_sum(entry, acc), do: {acc, entry + acc}

  def elixir_sum(entry, acc) do
    true = Process.get(Nx.Defn.Compiler) in [Nx.Defn.Evaluator, Nx.Defn.Identity]
    {acc, Nx.add(entry, acc)}
  end

  test "runs defn stream" do
    %_{} = stream = Nx.Defn.stream(&defn_sum/2, [0, 0])
    assert Nx.Stream.send(stream, 1) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(0)

    assert Nx.Stream.send(stream, 2) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(1)

    assert Nx.Stream.done(stream) == Nx.tensor(3)
  end

  test "runs elixir stream" do
    %_{} = stream = Nx.Defn.stream(&elixir_sum/2, [0, 0])
    assert Nx.Stream.send(stream, 1) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(0)

    assert Nx.Stream.send(stream, 2) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(1)

    assert Nx.Stream.done(stream) == Nx.tensor(3)
  end

  @tag :capture_log
  test "raises on errors" do
    Process.flag(:trap_exit, true)
    assert %_{} = stream = Nx.Defn.stream(fn _, _ -> :bar end, [1, 2])

    assert Nx.Stream.send(stream, 1) == :ok
    assert catch_exit(Nx.Stream.recv(stream))

    ref = Process.monitor(stream.pid)
    assert_receive {:DOWN, ^ref, _, _, _}
  end

  test "converts accumulator to tensors" do
    assert %_{} = stream = Nx.Defn.stream(fn -> :unused end, [1, {2, 3}])
    assert Nx.Stream.done(stream) == {Nx.tensor(2), Nx.tensor(3)}
  end

  test "raises if stream is not compatible on send" do
    assert %_{} = stream = Nx.Defn.stream(fn -> :unused end, [1, {2, 3}])

    assert_raise ArgumentError,
                 ~r/Nx stream expected a tensor of type, shape, and names on send/,
                 fn ->
                   Nx.Stream.send(stream, Nx.iota({3}))
                 end
  end

  test "raises if stream is not compatible on recv" do
    assert %_{} = stream = Nx.Defn.stream(fn _a, {b, c} -> {b, c} end, [1, {2, 3}])

    assert Nx.Stream.send(stream, Nx.iota({})) == :ok

    assert_raise ArgumentError,
                 ~r/Nx stream expected a tensor of type, shape, and names on recv/,
                 fn ->
                   Nx.Stream.recv(stream)
                 end
  end

  test "raises if already done" do
    assert %_{} = stream = Nx.Defn.stream(fn -> :bad end, [1, 2])
    assert Nx.Stream.done(stream) == Nx.tensor(2)
    assert {:noproc, _} = catch_exit(Nx.Stream.done(stream))
  end

  defn stream_iota(_, _), do: {Nx.iota({}), Nx.iota({})}

  @tag :capture_log
  test "uses the default backend on iota" do
    Process.flag(:trap_exit, true)
    args = [Nx.tensor(1), Nx.tensor(2)]
    Nx.default_backend(UnknownBackend)
    assert %_{} = stream = Nx.Defn.stream(&stream_iota/2, args)
    assert Nx.Stream.send(stream, hd(args))
    assert_receive {:EXIT, _, {:undef, _}}
  end
end
