defmodule EXLA.DefnAPITest do
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
                   "Invalid buffer passed to Execute() as argument 1 to replica 0: Invalid argument: Buffer has been deleted or donated.",
                   fn -> add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor) end

      assert_raise RuntimeError,
                   "CopyToHostAsync() called on deleted or donated buffer",
                   fn -> Nx.backend_transfer(tensor) end
    end
  end

  describe "stream" do
    defn defn_sum(entry, acc), do: {acc, entry + acc}

    # TODO: Test returning an empty map/tuple
    # TODO: Make infeed/outfeed dirty NIFs (cpu for host, io for others)

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
      # TODO: Build outfeed
      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.recv(stream) == [<<0::64-native>>]

      assert Nx.Stream.send(stream, 2) == :ok
      assert Nx.Stream.recv(stream) == [<<1::64-native>>]

      assert Nx.Stream.done(stream) == Nx.tensor(3)
    end

    test "send x2/recv x2" do
      %_{} = stream = EXLA.stream(&defn_sum/2, [0, 0])
      assert Nx.Stream.send(stream, 1) == :ok
      assert Nx.Stream.send(stream, 2) == :ok

      assert Nx.Stream.recv(stream) == [<<0::64-native>>]
      assert Nx.Stream.recv(stream) == [<<1::64-native>>]

      assert Nx.Stream.done(stream) == Nx.tensor(3)
    end
  end
end
