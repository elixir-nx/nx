defmodule EXLA.DeviceBackendTest do
  use ExUnit.Case, async: true

  test "transfers data" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, EXLA.DeviceBackend)
    assert %EXLA.DeviceBackend{state: {ref, :default}} = et.data
    assert is_reference(ref)

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, "CopyToHostAsync() called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "transfers data to a specific device" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, {EXLA.DeviceBackend, device_id: 0})
    assert %EXLA.DeviceBackend{state: {ref, :default}} = et.data
    assert is_reference(ref)

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, "CopyToHostAsync() called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "copies data" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, EXLA.DeviceBackend)
    assert %EXLA.DeviceBackend{state: {ref, :default}} = et.data
    assert is_reference(ref)

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "can be inspected" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.DeviceBackend)

    assert inspect(t) =~ ~r"""
           #Nx.Tensor<
             s64\[4\]
             EXLA.DeviceBackend<\{#Reference<[\d\.]+>, :default}>
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
