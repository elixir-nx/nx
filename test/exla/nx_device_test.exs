defmodule Exla.NxDeviceTest do
  use ExUnit.Case, async: true

  describe "nx device" do
    test "transfers data from nx<->device" do
      t = Nx.tensor([1, 2, 3, 4])

      et = Nx.device_transfer(t, Exla.NxDevice)
      assert {Exla.NxDevice, {ref, :default}} = et.data
      assert is_reference(ref)

      assert_raise ArgumentError,
                   ~r"cannot read Nx.Tensor data because the data is allocated on device Exla.NxDevice",
                   fn -> Nx.Util.to_binary(et) end

      nt = Nx.device_transfer(et)

      assert Nx.Util.to_binary(nt) ==
               <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

      assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn ->
        Nx.device_transfer(et)
      end
    end

    test "multiple reads and deallocation" do
      t = Nx.tensor([1, 2, 3, 4])

      et = Nx.device_transfer(t, Exla.NxDevice, key: :tensor)
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
                   ~r"could not find Exla client named :unknown",
                   fn ->
                     Nx.device_transfer(Nx.tensor([1, 2]), Exla.NxDevice, client: :unknown)
                   end
    end
  end

  describe "sharded nx device" do
    @describetag :multi_device

    test "transfers data from nx<->device" do
      t = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])

      et = Nx.device_transfer(t, Exla.ShardedNxDevice)
      assert {Exla.ShardedNxDevice, [buffer1, buffer2]} = et.data
      assert {ref1, :default} = buffer1.ref
      assert {ref2, :default} = buffer2.ref
      assert is_reference(ref1)
      assert is_reference(ref2)

      assert_raise ArgumentError,
                   ~r"cannot read Nx.Tensor data because the data is allocated on device Exla.ShardedNxDevice",
                   fn -> Nx.Util.to_binary(et) end

      nt = Nx.device_transfer(et)

      assert Nx.Util.to_binary(nt) ==
               <<1::64-native, 2::64-native, 3::64-native, 4::64-native, 1::64-native,
                 2::64-native, 3::64-native, 4::64-native>>

      assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn ->
        Nx.device_transfer(et)
      end
    end

    test "multiple reads and deallocation" do
      t = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])

      et = Nx.device_transfer(t, Exla.ShardedNxDevice, key: :tensor)
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
                   ~r"could not find Exla client named :unknown",
                   fn ->
                     Nx.device_transfer(Nx.tensor([1, 2]), Exla.ShardedNxDevice, client: :unknown)
                   end
    end
  end
end
