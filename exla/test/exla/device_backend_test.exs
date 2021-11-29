defmodule EXLA.DeviceBackendTest do
  use ExUnit.Case, async: true

  test "Nx.to_binary/1" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.DeviceBackend)
    assert Nx.to_binary(t) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
    assert Nx.to_binary(t, limit: 2) == <<1::64-native, 2::64-native>>
    assert Nx.to_binary(t, limit: 6) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "Nx.backend_transfer/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, {EXLA.DeviceBackend, device_id: 0})
    assert %EXLA.DeviceBackend{buffer: %EXLA.Buffer{}} = et.data

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "Nx.backend_transfer/2" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, EXLA.DeviceBackend)
    assert %EXLA.DeviceBackend{buffer: %EXLA.Buffer{} = old_buffer} = et.data

    # Transferring to the same device is a no-op
    et = Nx.backend_transfer(et, EXLA.DeviceBackend)
    assert %EXLA.DeviceBackend{buffer: new_buffer} = et.data
    assert old_buffer == new_buffer

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "Nx.backend_copy/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, EXLA.DeviceBackend)
    assert %EXLA.DeviceBackend{buffer: %EXLA.Buffer{} = old_buffer} = et.data

    # Copy to the same client/device_id still makes a copy
    et = Nx.backend_copy(t, EXLA.DeviceBackend)
    assert %EXLA.DeviceBackend{buffer: %EXLA.Buffer{} = new_buffer} = et.data
    assert old_buffer != new_buffer

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "Kernel.inspect/2" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.DeviceBackend)
    client = EXLAHelpers.client()

    assert inspect(t) =~ ~r"""
           #Nx.Tensor<
             s64\[4\]
             EXLA.DeviceBackend<#{client.name}:#{client.default_device_id}, [\d\.]+>
           >\
           """
  end

  test "raises on most operations" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.DeviceBackend)
    assert_raise RuntimeError, fn -> Nx.exp(t) end
  end

  test "raises on invalid client" do
    assert_raise ArgumentError,
                 ~r"could not find EXLA client named :unknown",
                 fn ->
                   Nx.backend_transfer(Nx.tensor([1, 2]), {EXLA.DeviceBackend, client: :unknown})
                 end
  end
end
