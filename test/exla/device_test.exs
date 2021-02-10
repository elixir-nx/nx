defmodule EXLA.DeviceTest do
  use ExUnit.Case, async: true

  describe "nx device" do
    test "transfers data from nx<->device" do
      t = Nx.tensor([1, 2, 3, 4])

      et = Nx.device_transfer(t, EXLA.Device)
      assert %Nx.BinaryTensor{device: EXLA.Device, state: {ref, :default}} = et.data
      assert is_reference(ref)

      assert_raise ArgumentError,
                   ~r"cannot read Nx.Tensor data because the data is allocated on device EXLA.Device",
                   fn -> Nx.to_binary(et) end

      nt = Nx.device_transfer(et)

      assert Nx.to_binary(nt) ==
               <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

      assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn ->
        Nx.device_transfer(et)
      end
    end

    test "multiple reads and deallocation" do
      t = Nx.tensor([1, 2, 3, 4])

      et = Nx.device_transfer(t, EXLA.Device, key: :tensor)
      assert Nx.device_read(et) == t
      assert Nx.device_read(et) == t
      assert Nx.device_deallocate(et) == :ok

      assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn ->
        Nx.device_read(et)
      end

      assert Nx.device_deallocate(et) == :already_deallocated
    end

    test "raises on invalid client" do
      assert_raise ArgumentError,
                   ~r"could not find EXLA client named :unknown",
                   fn ->
                     Nx.device_transfer(Nx.tensor([1, 2]), EXLA.Device, client: :unknown)
                   end
    end
  end
end
