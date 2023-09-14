defmodule EXLA.BackendTest do
  use EXLA.Case, async: true

  import Nx, only: [sigil_V: 2]

  setup do
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  @precision_error_doctests [
    expm1: 1,
    erfc: 1,
    erf: 1,
    sin: 1,
    cos: 1,
    tan: 1,
    cosh: 1,
    tanh: 1,
    asin: 1,
    asinh: 1,
    atanh: 1,
    ceil: 1,
    sigmoid: 1,
    fft: 2,
    ifft: 2,
    logsumexp: 2,
    conjugate: 1
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

  case EXLAHelpers.client() do
    %EXLA.Client{platform: :cuda} ->
      @precision_error_doctests [
                                  standard_deviation: 2,
                                  rsqrt: 1,
                                  acos: 1,
                                  variance: 2,
                                  atan2: 2,
                                  weighted_mean: 3,
                                  cbrt: 1
                                ] ++ @precision_error_doctests
      @inherently_unsupported_doctests [conv: 3] ++ @inherently_unsupported_doctests

    _ ->
      nil
  end

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

  describe "Nx.backend_copy/1" do
    test "same client" do
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

    test "different clients" do
      t = Nx.tensor([1, 2, 3, 4])

      et = Nx.backend_transfer(t, EXLA.Backend)
      assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = old_buffer} = et.data

      et = Nx.backend_copy(t, {EXLA.Backend, client: :other_host})
      assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = new_buffer} = et.data
      assert old_buffer != new_buffer
      assert new_buffer.client_name == :other_host
      assert new_buffer.device_id == 0

      nt = Nx.backend_copy(et)
      assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

      nt = Nx.backend_copy(et)
      assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
    end
  end

  test "Nx.pad/3 with multiple backends" do
    t = Nx.tensor([1, 1], backend: Nx.BinaryBackend)
    pad_value = Nx.tensor(0, backend: EXLA.Backend)
    result = Nx.pad(t, pad_value, [{1, 1, 0}])
    assert_equal(result, Nx.tensor([0, 1, 1, 0]))
  end

  test "Nx.LinAlg.svd/2" do
    t = Nx.iota({4, 4})
    assert {u, s, vt} = Nx.LinAlg.svd(t, max_iter: 10_000)

    reconstructed = u |> Nx.multiply(s) |> Nx.dot(vt)
    assert_all_close(t, reconstructed, atol: 1.0e-5)
  end

  test "multi-client" do
    a = Nx.tensor(1, backend: {EXLA.Backend, client: :host})
    b = Nx.tensor(2, backend: {EXLA.Backend, client: :other_host})
    assert_equal(Nx.add(a, b), Nx.tensor(3))
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
    defn double(fun), do: double_transform(fun.())

    deftransformp(double_transform(x), do: Nx.backend_transfer(Nx.Defn.Kernel.*(x, x)))

    test "invokes from within defn" do
      t = Nx.tensor(11)
      assert double(fn -> t end) |> Nx.to_number() == 121
    end

    test "raises on invalid client" do
      assert_raise ArgumentError,
                   ~r"could not find EXLA client named :unknown",
                   fn ->
                     Nx.backend_transfer(Nx.tensor([1, 2]), {EXLA.Backend, client: :unknown})
                   end
    end
  end

  test "conjugate" do
    assert inspect(Nx.conjugate(~V[1 2-0i 3+0i 0-i 0-2i])) =~
             "1.0-0.0i, 2.0+0.0i, 3.0-0.0i, 0.0+1.0i, 0.0+2.0i"
  end
end
