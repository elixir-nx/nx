defmodule Exla.NxDeviceTest do
  use ExUnit.Case, async: true

  test "transfers data from nx<->device" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.device_transfer(t, Exla.NxDevice)
    assert {Exla.NxDevice, {ref, :default}} = et.data
    assert is_reference(ref)

    assert_raise ArgumentError,
                 ~r"cannot read Nx.Tensor data because the data is allocated on device Exla.NxDevice",
                 fn -> Nx.Util.to_bitstring(et) end

    nt = Nx.device_transfer(et)
    assert Nx.Util.to_bitstring(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn -> Nx.device_transfer(et) end
  end

  test "multiple reads and deallocation" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.device_transfer(t, Exla.NxDevice, key: :tensor)
    assert Nx.device_read(et) == t
    assert Nx.device_read(et) == t
    assert Nx.device_deallocate(et) == :ok

    assert_raise RuntimeError, "Attempt to read from deallocated buffer.", fn -> Nx.device_read(et) end
    assert Nx.device_deallocate(et) == :already_deallocated
  end

  test "raises on invalid client" do
    assert_raise ArgumentError,
                 ~r"could not find Exla client named :unknown",
                 fn -> Nx.device_transfer(Nx.tensor([1, 2]), Exla.NxDevice, client: :unknown) end
  end
end
