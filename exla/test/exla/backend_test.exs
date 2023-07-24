defmodule EXLA.BackendTest do
  use EXLA.Case, async: true

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
    logsumexp: 2
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
    defn(double(fun), do: double_transform(fun.()))

    deftransformp(double_transform(x), do: Nx.backend_transfer(Nx.Defn.Kernel.*(x, x)))

    test "invokes from within defn" do
      t = Nx.tensor(11)
      assert double(fn -> t end) |> Nx.to_number() == 121
    end
  end

  test "raises on invalid client" do
    assert_raise ArgumentError,
                 ~r"could not find EXLA client named :unknown",
                 fn ->
                   Nx.backend_transfer(Nx.tensor([1, 2]), {EXLA.Backend, client: :unknown})
                 end
  end

  test "Nx.logexpsum/1" do
    assert_equal(Nx.logsumexp(Nx.tensor([1, 2, 3, 4, 5, 6])), Nx.tensor(6.456193447113037))

    assert_equal(
      Nx.logsumexp(Nx.tensor([1, 2, 3, 4, 5, 6]), exp_scaling_factor: 0.5),
      Nx.tensor(5.7630462646484375)
    )

    assert_equal(
      Nx.logsumexp(Nx.tensor([1, 2, 3, 4, 5, 6]),
        exp_scaling_factor: Nx.tensor([-1, -1, -1, 1, 1, 1])
      ),
      Nx.tensor(6.356536865234375)
    )

    assert_equal(Nx.logsumexp(Nx.tensor([[1, 2], [3, 4], [5, 6]])), Nx.tensor(6.456193447113037))

    assert_equal(
      Nx.logsumexp(Nx.tensor([[1, 2], [3, 4], [5, 6]], names: [:x, :y]), axes: [:x]),
      Nx.tensor([5.1429314613342285, 6.1429314613342285], names: [:y])
    )

    assert_equal(
      Nx.logsumexp(Nx.tensor([[1, 2], [3, 4], [5, 6]], names: [:x, :y]), axes: [:y]),
      Nx.tensor([2.3132615089416504, 4.31326150894165, 6.31326150894165], names: [:x])
    )

    assert_equal(
      Nx.logsumexp(Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], names: [:x, :y, :z]),
        axes: [:x, :z]
      ),
      Nx.tensor([6.331411361694336, 8.331411361694336], names: [:y])
    )

    assert_equal(
      Nx.logsumexp(Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], names: [:x, :y, :z]),
        axes: [:x, :z],
        keep_axes: true
      ),
      Nx.tensor(
        [
          [
            [6.331411361694336],
            [8.331411361694336]
          ]
        ],
        names: [:x, :y, :z]
      )
    )

    assert_equal(
      Nx.logsumexp(Nx.vectorize(Nx.tensor([[1, 2], [3, 4], [5, 6]]), :x),
        axes: [0],
        keep_axes: true
      ),
      Nx.vectorize(
        Nx.tensor([[2.3132615089416504], [4.31326150894165], [6.31326150894165]]),
        :x
      )
    )
  end
end
