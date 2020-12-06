defmodule Nx.DeviceTest do
  use ExUnit.Case, async: true

  test "transfers data from nx<->device" do
    t = Nx.tensor([1, 2, 3, 4])

    pt = Nx.device_transfer(t, Nx.ProcessDevice, key: :tensor)
    assert pt.data == {Nx.ProcessDevice, :tensor}
    assert Process.get(:tensor)

    assert_raise ArgumentError,
                 ~r"cannot read Nx.Tensor data because the data is allocated on device Nx.ProcessDevice",
                 fn -> Nx.Util.to_bitstring(pt) end

    nt = Nx.device_transfer(pt)
    assert Nx.Util.to_bitstring(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
    refute Process.get(:tensor)

    assert_raise RuntimeError, "deallocated", fn -> Nx.device_transfer(pt) end
  end

  test "multiple reads and deallocation" do
    t = Nx.tensor([1, 2, 3, 4])

    # The default device always returns true for deallocating
    # because it is garbage collector dependent. If the user
    # does not use it, then it works.
    assert Nx.device_deallocate(t) == :ok
    assert Nx.device_deallocate(t) == :ok

    pt = Nx.device_transfer(t, Nx.ProcessDevice, key: :tensor)
    assert Nx.device_read(pt) == t
    assert Nx.device_read(pt) == t
    assert Nx.device_deallocate(pt) == :ok

    assert_raise RuntimeError, "deallocated", fn -> Nx.device_read(pt) end
    assert Nx.device_deallocate(pt) == :already_deallocated
  end

  test "raises when transferring between devices" do
    pt = Nx.device_transfer(Nx.tensor([1, 2, 3, 4]), Nx.ProcessDevice, key: :tensor)

    assert_raise ArgumentError,
                 ~r"cannot transfer from Nx.ProcessDevice to UnknownDevice",
                 fn -> Nx.device_transfer(pt, UnknownDevice, :whatever) end
  end
end
