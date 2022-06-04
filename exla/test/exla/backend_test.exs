defmodule EXLA.BackendTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  @precision_error_doctests [
    expm1: 1,
    erfc: 1,
    erf: 1,
    cosh: 1,
    tanh: 1,
    asin: 1,
    asinh: 1,
    atanh: 1,
    ceil: 1,
    sigmoid: 1,
    fft: 2
  ]

  @temporarily_broken_doctests [
    # XLA currently doesn't support complex conversion
    as_type: 2
  ]

  @inherently_unsupported_doctests [
    # XLA requires signed and unsigned tensors to be at least of size 32
    random_uniform: 4
  ]

  @unrelated_doctests [
    default_backend: 1
  ]

  doctest Nx,
    except:
      [:moduledoc] ++
        @precision_error_doctests ++
        @temporarily_broken_doctests ++
        @inherently_unsupported_doctests ++
        @unrelated_doctests

  test "Nx.to_binary/1" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.Backend)
    assert Nx.to_binary(t) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
    assert Nx.to_binary(t, limit: 2) == <<1::64-native, 2::64-native>>
    assert Nx.to_binary(t, limit: 6) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "Nx.backend_transfer/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, {EXLA.Backend, device_id: 0})
    assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{}} = et.data

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "Nx.backend_transfer/2" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, EXLA.Backend)
    assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = old_buffer} = et.data

    # Transferring to the same device is a no-op
    et = Nx.backend_transfer(et, EXLA.Backend)
    assert %EXLA.Backend{buffer: new_buffer} = et.data
    assert old_buffer == new_buffer

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "Nx.backend_copy/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, EXLA.Backend)
    assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = old_buffer} = et.data

    # Copy to the same client/device_id still makes a copy
    et = Nx.backend_copy(t, EXLA.Backend)
    assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = new_buffer} = et.data
    assert old_buffer != new_buffer

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "Kernel.inspect/2" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.Backend)

    assert inspect(t) ==
             """
             #Nx.Tensor<
               s64[4]
               [1, 2, 3, 4]
             >\
             """
  end

  describe "within JIT" do
    import Nx.Defn

    # This is not really meant to work in practice,
    # but it does work with the Nx.BinaryBackend so
    # we make it work for EXLA too.
    defn double(x \\ 0), do: x * x

    test "invokes from within defn" do
      assert double(Nx.tensor(11)) |> Nx.to_number() == 121
    end
  end

  test "raises on invalid client" do
    assert_raise ArgumentError,
                 ~r"could not find EXLA client named :unknown",
                 fn ->
                   Nx.backend_transfer(Nx.tensor([1, 2]), {EXLA.Backend, client: :unknown})
                 end
  end
end
