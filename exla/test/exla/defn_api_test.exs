defmodule EXLA.DefnAPITest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler {EXLA, max_float_type: {:f, 64}}

  describe "async" do
    defn add_two_async(a, b), do: a + b

    test "awaits" do
      async = Nx.Defn.async(&add_two_async/2, [1, 2], EXLA)
      assert Nx.Async.await!(async) == Nx.tensor(3)
    end

    test "awaits with keep_on_device" do
      async = Nx.Defn.async(&add_two_async/2, [1, 2], EXLA, run_options: [keep_on_device: true])
      assert %Nx.Tensor{} = tensor = Nx.Async.await!(async)
      assert %Nx.BinaryBackend{device: EXLA.Device, state: {ref, :default}} = tensor.data
      assert is_reference(ref)
      assert tensor |> Nx.device_read() |> Nx.to_binary() == <<3::64-native>>
    end

    # TODO: We need to track this on the device
    # test "cannot await twice" do
    #   async = Nx.Defn.async(&add_two_async/2, [1, 2], EXLA)
    #   assert Nx.Async.await!(async) == Nx.tensor(3)
    #   assert_raise RuntimeError, fn -> Nx.Async.await!(async) end
    # end
  end

  describe "options" do
    @defn_compiler {EXLA, run_options: [keep_on_device: true]}
    defn add_two_keep_on_device(a, b), do: a + b

    test "keeps data on device" do
      tensor = add_two_keep_on_device(1, 2)
      assert %Nx.BinaryBackend{device: EXLA.Device, state: {ref, :default}} = tensor.data
      assert is_reference(ref)
      assert tensor |> Nx.device_read() |> Nx.to_binary() == <<3::64-native>>

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert %Nx.BinaryBackend{device: EXLA.Device, state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      assert tensor |> Nx.device_read() |> Nx.to_binary() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert tensor |> Nx.device_transfer() |> Nx.to_binary() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   "Attempt to read from deallocated buffer.",
                   fn -> Nx.device_read(tensor) end
    end
  end
end
