defmodule EXLA.Defn.APITest do
  use ExUnit.Case, async: true

  import Nx.Defn

  setup do
    Nx.Defn.default_options(compiler: EXLA)
    :ok
  end

  describe "options" do
    defn add_two_keep_on_device(a, b), do: a + b

    # Ignored logged errors, since they are expected
    @tag capture_log: true
    test "keeps data on device" do
      Nx.Defn.default_options(compiler: EXLA, run_options: [keep_on_device: true])

      tensor = add_two_keep_on_device(1, 2)
      assert %EXLA.DeviceBackend{buffer: %EXLA.Buffer{}} = tensor.data

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert %EXLA.DeviceBackend{buffer: %EXLA.Buffer{}} = tensor.data

      assert tensor |> Nx.backend_transfer() |> Nx.to_binary() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   ~r"Buffer has been deleted or donated",
                   fn -> add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor) end

      assert_raise RuntimeError,
                   ~r"called on deleted or donated buffer",
                   fn -> Nx.backend_transfer(tensor) end
    end
  end

  describe "containers" do
    defn container_as_input(%Container{a: a, b: b}) do
      a * b
    end

    defn update_container(var, x) do
      %{var | b: x}
    end

    defn dot_container(container) do
      container.a * container.b
    end

    defn container_with_map_tuple(%Container{a: {x, y}, b: %{} = b}) do
      %{a: a, b: b} = b
      x * y * a * b
    end

    test "matched as input" do
      inp = %Container{a: Nx.tensor(5), b: Nx.tensor(3)}

      assert container_as_input(inp) == Nx.tensor(15)
    end

    test "updated" do
      inp = %Container{a: Nx.tensor(1), b: 2, c: :reset, d: :keep}

      assert %Container{a: a, b: b, c: c, d: d} = update_container(inp, Nx.tensor(8))
      assert a == Nx.tensor(1)
      assert b == Nx.tensor(8)
      assert c == nil
      assert d == :keep
    end

    test "can be used with dot syntax" do
      inp = %Container{a: Nx.tensor(1), b: 2}
      assert dot_container(inp) == Nx.tensor(2)
    end

    test "can be used with nested collections" do
      inp = %Container{a: {Nx.tensor(1), Nx.tensor(2)}, b: %{a: Nx.tensor(3), b: Nx.tensor(4)}}
      assert container_with_map_tuple(inp) == Nx.tensor(24)
    end
  end

  describe "stream" do
    defn defn_sum(entry, acc), do: {acc, entry + acc}

    test "immediately done" do
      stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.done(stream) == Nx.tensor(0)

      stream = EXLA.stream(&defn_sum/2, [1, 2])
      assert Nx.Stream.done(stream) == Nx.tensor(2)
    end

    test "immediately done with keep_on_device" do
      stream = EXLA.stream(&defn_sum/2, [0, 0], run_options: [keep_on_device: true])
      assert %Nx.Tensor{data: %EXLA.DeviceBackend{}} = done = Nx.Stream.done(stream)
      assert Nx.backend_transfer(done) == Nx.tensor(0)

      stream = EXLA.stream(&defn_sum/2, [1, 2], run_options: [keep_on_device: true])
      assert %Nx.Tensor{data: %EXLA.DeviceBackend{}} = done = Nx.Stream.done(stream)
      assert Nx.backend_transfer(done) == Nx.tensor(2)
    end

    test "send/recv" do
      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(0)

      assert Nx.Stream.send(stream, 2) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(1)

      assert Nx.Stream.done(stream) == Nx.tensor(3)
    end

    test "send x2/recv x2" do
      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.send(stream, 2) == :ok

      assert Nx.Stream.recv(stream) == Nx.tensor(0)
      assert Nx.Stream.recv(stream) == Nx.tensor(1)

      assert Nx.Stream.done(stream) == Nx.tensor(3)
    end

    defn stream_composite(i, {a, {b, c}}) do
      a = a + i
      b = b * i
      c = Nx.power(c, i)
      {{{a, b}, c}, {a, {b, c}}}
    end

    test "send/recv with composite types" do
      %_{} = stream = EXLA.stream(&stream_composite/2, [0, {0, {1, 2}}])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == {{Nx.tensor(1), Nx.tensor(1)}, Nx.tensor(2)}

      assert Nx.Stream.send(stream, 2) == :ok
      assert Nx.Stream.recv(stream) == {{Nx.tensor(3), Nx.tensor(2)}, Nx.tensor(4)}

      assert Nx.Stream.done(stream) == {Nx.tensor(3), {Nx.tensor(2), Nx.tensor(4)}}
    end

    defn stream_empty_outfeed(i, t), do: {{}, i + t}

    test "send/recv with empty outfeed" do
      %_{} = stream = EXLA.stream(&stream_empty_outfeed/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == {}

      assert Nx.Stream.send(stream, 2) == :ok
      assert Nx.Stream.recv(stream) == {}

      assert Nx.Stream.done(stream) == Nx.tensor(3)
    end

    defn stream_empty_acc(i, {}), do: {i * i, {}}

    test "send/recv with empty acc" do
      %_{} = stream = EXLA.stream(&stream_empty_acc/2, [0, {}])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(1)

      assert Nx.Stream.send(stream, 2) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(4)

      assert Nx.Stream.done(stream) == {}
    end

    test "handles failure before writing" do
      {_, ref} = spawn_monitor(fn -> EXLA.stream(&defn_sum/2, [0, 0]) end)
      assert_receive {:DOWN, ^ref, _, _, _}

      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(0)
      assert Nx.Stream.done(stream) == Nx.tensor(1)
    end

    test "handles failure after writing" do
      {_, ref} =
        spawn_monitor(fn ->
          stream = EXLA.stream(&defn_sum/2, [0, 0])
          assert Nx.Stream.send(stream, 1) == :ok
        end)

      assert_receive {:DOWN, ^ref, _, _, _}

      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(0)
      assert Nx.Stream.done(stream) == Nx.tensor(1)
    end

    test "raises if recv is pending on done" do
      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok

      assert_raise RuntimeError,
                   "cannot mark stream as done when there are recv messages pending",
                   fn -> Nx.Stream.done(stream) end
    end

    test "raises if stream is done when recving" do
      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.done(stream) == Nx.tensor(0)

      assert_raise RuntimeError,
                   "cannot recv from stream because it has been terminated",
                   fn -> Nx.Stream.recv(stream) end
    end

    defn container_stream(%Container{a: a} = elem, %Container{b: b} = acc) do
      {%{elem | a: a + b}, %{acc | b: a + b}}
    end

    test "container in and out" do
      args = [%Container{a: 0, b: 0, c: :reset, d: :elem}, %Container{a: 0, b: 0, d: :acc}]
      %_{} = stream = EXLA.stream(&container_stream/2, args)

      assert Nx.Stream.send(stream, %Container{a: 1, b: -1}) == :ok
      assert Nx.Stream.recv(stream) == %Container{a: Nx.tensor(1), b: Nx.tensor(-1), d: :elem}

      assert Nx.Stream.send(stream, %Container{a: 2, b: -2}) == :ok
      assert Nx.Stream.recv(stream) == %Container{a: Nx.tensor(3), b: Nx.tensor(-2), d: :elem}

      assert Nx.Stream.done(stream) == %Container{a: Nx.tensor(0), b: Nx.tensor(3), d: :acc}
    end
  end

  describe "hooks" do
    require Logger

    defp send_to_self(tag) do
      parent = self()
      fn value -> send(parent, {tag, value}) end
    end

    defn hook_default(a, b) do
      hook(a + b, :default, &Logger.error("add: #{inspect(&1)}"))
    end

    test "executes hook with default" do
      assert ExUnit.CaptureLog.capture_log(fn -> hook_default(2, 3) end) =~
               """
               add: #Nx.Tensor<
                 s64
                 5
               >
               """
    end

    test "executes hook with callback" do
      assert EXLA.jit(&hook_default/2, [2, 3], hooks: %{default: send_to_self(:tag)}) ==
               Nx.tensor(5)

      assert_receive {:tag, tensor}
      assert tensor == Nx.tensor(5)
    end

    defn hook_optional(a, b) do
      hook(a + b, :optional)
    end

    test "executes optional hook" do
      assert hook_optional(2, 3) == Nx.tensor(5)

      assert EXLA.jit(&hook_optional/2, [2, 3], hooks: %{optional: send_to_self(:tag)}) ==
               Nx.tensor(5)

      assert_receive {:tag, tensor}
      assert tensor == Nx.tensor(5)
    end

    defn hook_factorial(x) do
      {factorial, _} =
        while {factorial = 1.0, x}, Nx.greater(x, 1) do
          hook({factorial * x, x - 1}, :factorial)
        end

      factorial
    end

    test "executes hook within while" do
      assert EXLA.jit(&hook_factorial/1, [5], hooks: %{factorial: send_to_self(:tag)}) ==
               Nx.tensor(120.0)

      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(5.0), Nx.tensor(4)}
      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(20.0), Nx.tensor(3)}
      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(60.0), Nx.tensor(2)}
      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(120.0), Nx.tensor(1)}
    end

    defn hook_cond(a, b) do
      cond do
        a == -1 -> hook(b * 2, :cond)
        a == 1 -> hook(b / 2, :cond)
        true -> hook(Nx.power(b, 2), :cond)
      end
    end

    test "executes hook within cond" do
      assert EXLA.jit(&hook_cond/2, [1, 4], hooks: %{cond: send_to_self(:tag)}) ==
               Nx.tensor(2.0)

      assert_received {:tag, tensor}
      assert tensor == Nx.tensor(2.0)

      assert EXLA.jit(&hook_cond/2, [-1, 4], hooks: %{cond: send_to_self(:tag)}) ==
               Nx.tensor(8.0)

      assert_received {:tag, tensor}
      assert tensor == Nx.tensor(8)

      assert EXLA.jit(&hook_cond/2, [0, 4], hooks: %{cond: send_to_self(:tag)}) ==
               Nx.tensor(16.0)

      assert_received {:tag, tensor}
      assert tensor == Nx.tensor(16)
    end

    defn hook_container(container) do
      hook(container, :container)
    end

    test "executes hook with container" do
      container = %Container{a: 1, b: 2, c: :reset, d: :elem}
      EXLA.jit(&hook_container/1, [container], hooks: %{container: send_to_self(:tag)})

      assert_receive {:tag, %Container{a: a, b: b, c: nil, d: :elem}}
      assert a == Nx.tensor(1)
      assert b == Nx.tensor(2)
    end

    defn hook_stream(entry, acc), do: hook({acc, entry + acc}, :stream)

    test "executes hook with stream" do
      %_{} = stream = EXLA.stream(&hook_stream/2, [0, 0], hooks: %{stream: send_to_self(:tag)})
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(0)
      assert_receive {:tag, {previous_acc, new_acc}}
      assert previous_acc == Nx.tensor(0)
      assert new_acc == Nx.tensor(1)
      refute_received _

      assert Nx.Stream.send(stream, 2) == :ok
      assert Nx.Stream.recv(stream) == Nx.tensor(1)
      assert_receive {:tag, {previous_acc, new_acc}}
      assert previous_acc == Nx.tensor(1)
      assert new_acc == Nx.tensor(3)
      refute_received _

      assert Nx.Stream.done(stream) == Nx.tensor(3)
      refute_received _
    end
  end
end
