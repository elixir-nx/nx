defmodule EXLA.Defn.APITest do
  use ExUnit.Case, async: true

  import Nx.Defn

  describe "options" do
    @defn_compiler {EXLA, run_options: [keep_on_device: true]}
    defn add_two_keep_on_device(a, b), do: a + b

    # Ignored logged errors, since they are expected
    @tag capture_log: true
    test "keeps data on device" do
      tensor = add_two_keep_on_device(1, 2)
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      assert tensor |> Nx.backend_transfer() |> Nx.to_binary() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   ~r"called on deleted or donated buffer",
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
end
