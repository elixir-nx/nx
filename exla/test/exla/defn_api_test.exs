defmodule EXLA.DefnAPITest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler EXLA

  describe "async" do
    defn add_two_async(a, b), do: a + b

    test "awaits" do
      async = Nx.Defn.async(&add_two_async/2, [1, 2], compiler: EXLA)
      assert Nx.Async.await!(async) == Nx.tensor(3)
    end

    test "awaits with keep_on_device" do
      async =
        Nx.Defn.async(&add_two_async/2, [1, 2],
          compiler: EXLA,
          run_options: [keep_on_device: true]
        )

      assert %Nx.Tensor{} = tensor = Nx.Async.await!(async)
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)
      assert tensor |> Nx.backend_transfer() |> Nx.to_binary() == <<3::64-native>>
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
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      assert tensor |> Nx.backend_transfer() |> Nx.to_binary() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   "Attempt to read from deallocated buffer.",
                   fn -> add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor) end

      assert_raise RuntimeError,
                   "Attempt to read from deallocated buffer.",
                   fn -> Nx.backend_transfer(tensor) end
    end
  end
end
