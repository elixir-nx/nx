defmodule Nx.Defn.StreamTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import ExUnit.CaptureLog

  defn defn_sum(entry, acc), do: {acc, entry + acc}

  def elixir_sum(entry, acc) do
    true = Process.get(Nx.Defn.Compiler) in [Nx.Defn.Evaluator, Nx.Defn.Debug]
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

  defn defn_sum_with_args(entry, acc, a, b), do: {acc, entry + acc + (a - b)}

  test "runs defn stream with args" do
    %_{} = stream = Nx.Defn.stream(&defn_sum_with_args/4, [0, 0, 2, 1])
    assert Nx.Stream.send(stream, 1) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(0)

    assert Nx.Stream.send(stream, 2) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(2)

    assert Nx.Stream.done(stream) == Nx.tensor(5)
  end

  test "runs elixir stream" do
    %_{} = stream = Nx.Defn.stream(&elixir_sum/2, [0, 0])
    assert Nx.Stream.send(stream, 1) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(0)

    assert Nx.Stream.send(stream, 2) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(1)

    assert Nx.Stream.done(stream) == Nx.tensor(3)
  end

  test "converts accumulator to tensors" do
    assert %_{} = stream = Nx.Defn.stream(fn _, _ -> {0, 0} end, [1, {2, 3}])
    assert Nx.Stream.done(stream) == {Nx.tensor(2), Nx.tensor(3)}
  end

  test "can recv before send" do
    %_{} = stream = Nx.Defn.stream(&defn_sum/2, [0, 0])
    task = Task.async(fn -> Nx.Stream.recv(stream) end)
    Process.sleep(100)
    assert Nx.Stream.send(stream, 1) == :ok
    assert Task.await(task) == Nx.tensor(0)
  end

  @tag :capture_log
  test "raises on errors" do
    Process.flag(:trap_exit, true)
    assert %_{} = stream = Nx.Defn.stream(fn _, _ -> 0 end, [1, 2])

    assert Nx.Stream.send(stream, 1) == :ok
    assert catch_exit(Nx.Stream.recv(stream))

    ref = Process.monitor(stream.pid)
    assert_receive {:DOWN, ^ref, _, _, _}
  end

  test "raises if stream is not compatible on send" do
    assert %_{} = stream = Nx.Defn.stream(fn _, _ -> {0, 0} end, [1, {2, 3}])

    assert_raise ArgumentError,
                 ~r/Nx stream expected a tensor of type, shape, and names on send/,
                 fn -> Nx.Stream.send(stream, Nx.iota({3})) end
  end

  test "raises if stream is not compatible on recv" do
    assert %_{} = stream = Nx.Defn.stream(fn _a, {b, c} -> {b, c} end, [1, {2, 3}])

    assert Nx.Stream.send(stream, Nx.iota({})) == :ok

    assert_raise ArgumentError,
                 ~r/Nx stream expected a tensor of type, shape, and names on recv/,
                 fn -> Nx.Stream.recv(stream) end
  end

  test "raises if already done" do
    assert %_{} = stream = Nx.Defn.stream(fn _, _ -> 0 end, [1, 2])
    assert Nx.Stream.done(stream) == Nx.tensor(2)
    assert {:noproc, _} = catch_exit(Nx.Stream.done(stream))
  end

  test "raises if recv is pending on done" do
    %_{} = stream = Nx.Defn.stream(&defn_sum/2, [0, 0])
    assert Nx.Stream.send(stream, 1) == :ok

    assert_raise RuntimeError,
                 "cannot mark stream as done when there are recv messages pending",
                 fn -> Nx.Stream.done(stream) end
  end

  test "raises if stream is done when recving" do
    Process.flag(:trap_exit, true)
    assert %_{} = stream = Nx.Defn.stream(fn _, _ -> 0 end, [1, 2])

    assert capture_log(fn ->
             Task.start_link(fn -> Nx.Stream.recv(stream) end)
             Process.sleep(100)
             Nx.Stream.done(stream)
             assert_receive {:EXIT, _, {%RuntimeError{}, _}}
           end) =~ "cannot recv from stream because it has been terminated"
  end

  defn stream_iota(_, _), do: {Nx.iota({}), Nx.iota({})}

  @tag :capture_log
  test "uses the default backend on iota" do
    Process.flag(:trap_exit, true)
    args = [Nx.tensor(1), Nx.tensor(2)]
    Nx.default_backend(ProcessBackend)
    assert %_{} = stream = Nx.Defn.stream(&stream_iota/2, args)
    assert Nx.Stream.send(stream, hd(args))
    assert_receive {:EXIT, _, {%RuntimeError{message: "not supported"}, _}}, 500
  end

  defn container_stream(%Container{a: a} = elem, %Container{b: b} = acc) do
    {%{elem | a: a + b}, %{acc | b: a + b}}
  end

  test "container in and out" do
    args = [%Container{a: 0, b: 0, c: :reset, d: :elem}, %Container{a: 0, b: 0, d: :acc}]
    %_{} = stream = Nx.Defn.stream(&container_stream/2, args)

    assert Nx.Stream.send(stream, %Container{a: 1, b: -1}) == :ok
    assert Nx.Stream.recv(stream) == %Container{a: Nx.tensor(1), b: Nx.tensor(-1), d: :elem}

    assert Nx.Stream.send(stream, %Container{a: 2, b: -2}) == :ok
    assert Nx.Stream.recv(stream) == %Container{a: Nx.tensor(3), b: Nx.tensor(-2), d: :elem}

    assert Nx.Stream.done(stream) == %Container{a: Nx.tensor(0), b: Nx.tensor(3), d: :acc}
  end

  defn lazy_container_stream(%LazyWrapped{a: a, c: c}, acc) do
    {acc, acc + a - c}
  end

  test "lazy container in" do
    args = [%LazyOnly{a: 0, b: 0, c: 0}, 0]
    %_{} = stream = Nx.Defn.stream(&lazy_container_stream/2, args)

    assert Nx.Stream.send(stream, %LazyOnly{a: 3, b: 0, c: -1}) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(0)

    assert Nx.Stream.send(stream, %LazyOnly{a: 5, b: 0, c: 2}) == :ok
    assert Nx.Stream.recv(stream) == Nx.tensor(4)

    assert Nx.Stream.done(stream) == Nx.tensor(7)
  end
end
