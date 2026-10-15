defmodule EXLA.BackendTest do
  use EXLA.Case, async: true

  import Nx, only: [sigil_VEC: 2]

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA, precision: :default)
    :ok
  end

  @excluded_doctests [
    asin: 1,
    atan: 1,
    tan: 1,
    acos: 1,
    cosh: 1,
    erf_inv: 1,
    erfc: 1,
    sinh: 1,
    atanh: 1,
    asinh: 1,
    logsumexp: 2,
    exp: 1,
    expm1: 1,
    rsqrt: 1,
    # Differ from the expected output in the last float digit on some platforms
    cbrt: 1,
    covariance: 3,
    log10: 1,
    # Doctests copied below as regular tests, so the ones that differ in the
    # last float digit on CUDA can be tagged :skip_cuda
    atan2: 2,
    phase: 1,
    standard_deviation: 2,
    variance: 2,
    weighted_mean: 3,
    reduce: 4
  ]

  doctest Nx,
    except: [:moduledoc] ++ @excluded_doctests

  describe "doctests that differ on CUDA" do
    # These doctests differ from the expected output in the last float digit
    # on CUDA, so they are tagged :skip_cuda.

    test "Nx.atan2/2" do
      neg_and_pos_zero_columns = Nx.tensor([[-0.0], [0.0]], type: :f64)
      neg_and_pos_zero_rows = Nx.tensor([-0.0, 0.0], type: :f64)

      assert_equal(
        Nx.atan2(neg_and_pos_zero_columns, neg_and_pos_zero_rows),
        Nx.tensor([[-3.141592653589793, -0.0], [3.141592653589793, 0.0]], type: :f64)
      )
    end

    @tag :skip_cuda
    test "Nx.atan2/2 between scalars" do
      assert_equal(Nx.atan2(1, 2), Nx.tensor(0.4636476))
    end

    @tag :skip_cuda
    test "Nx.atan2/2 between tensors and scalars" do
      assert_equal(
        Nx.atan2(Nx.tensor([1, 2, 3], names: [:data]), 1),
        Nx.tensor([0.7853982, 1.1071488, 1.2490457], names: [:data])
      )

      assert_equal(
        Nx.atan2(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data])),
        Nx.tensor([0.7853982, 0.4636476, 0.32175055], names: [:data])
      )
    end

    test "Nx.phase/1" do
      assert_equal(Nx.phase(1), Nx.tensor(0.0))
    end

    @tag :skip_cuda
    test "Nx.phase/1 of complex numbers" do
      assert_equal(Nx.phase(Complex.new(1, 2)), Nx.tensor(1.1071488))
      assert_equal(Nx.phase(~VEC[1+2i -2+1i]), Nx.tensor([1.1071488, 2.6779451]))
    end

    test "Nx.weighted_mean/3" do
      assert_equal(Nx.weighted_mean(Nx.tensor(42), Nx.tensor(2)), Nx.tensor(42.0))

      assert_equal(
        Nx.weighted_mean(Nx.tensor([1, 2, 3], names: [:x]), Nx.tensor([4, 5, 6]), axes: [0]),
        Nx.tensor(2.1333334)
      )

      assert_equal(
        Nx.weighted_mean(Nx.tensor([1, 2, 3], type: :u8, names: [:x]), Nx.tensor([1, 3, 5]),
          axes: [:x]
        ),
        Nx.tensor(2.4444444)
      )

      assert_equal(
        Nx.weighted_mean(Nx.iota({3, 4}), Nx.tensor([1, 2, 3, 4]), axes: [1]),
        Nx.tensor([2.0, 6.0, 10.0])
      )

      assert_equal(
        Nx.weighted_mean(Nx.iota({2, 4, 4, 1}), Nx.broadcast(2, {4, 4}), axes: [1, 2]),
        Nx.tensor([[7.5], [23.5]])
      )

      t = Nx.tensor([[1, 2, 3], [1, 1, 1]]) |> Nx.vectorize(:x)
      w = Nx.tensor([[1, 1, 1], [0, 0, 1]]) |> Nx.vectorize(:y)

      assert_equal(
        Nx.weighted_mean(t, w),
        Nx.tensor([[2.0, 3.0], [1.0, 1.0]]) |> Nx.vectorize([:x, :y])
      )
    end

    @tag :skip_cuda
    test "Nx.weighted_mean/3 with float rounding" do
      assert_equal(
        Nx.weighted_mean(Nx.tensor([1, 2, 3]), Nx.tensor([3, 2, 1])),
        Nx.tensor(1.6666666)
      )

      t = Nx.tensor(Nx.iota({2, 2, 3}), names: [:x, :y, :z])
      weights = Nx.tensor([[[0, 1, 2], [1, 1, 0]], [[-1, 1, -1], [1, 1, -1]]])

      assert_equal(
        Nx.weighted_mean(t, weights, axes: [-1], keep_axes: true),
        Nx.tensor([[[1.6666666], [3.5]], [[7.0], [8.0]]], names: [:x, :y, :z])
      )
    end

    test "Nx.variance/2" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert_equal(Nx.variance(t), Nx.tensor(1.25))
      assert_equal(Nx.variance(t, axes: [0]), Nx.tensor([1.0, 1.0]))
      assert_equal(Nx.variance(t, axes: [1]), Nx.tensor([0.25, 0.25]))
      assert_equal(Nx.variance(t, axes: [0], ddof: 1), Nx.tensor([2.0, 2.0]))
      assert_equal(Nx.variance(t, axes: [1], ddof: 1), Nx.tensor([0.5, 0.5]))
      assert_equal(Nx.variance(t, axes: [1], keep_axes: true), Nx.tensor([[0.25], [0.25]]))

      assert_equal(
        Nx.variance(Nx.tensor([[1, 2], [0, 4]]) |> Nx.vectorize(:x)),
        Nx.tensor([0.25, 4.0]) |> Nx.vectorize(:x)
      )
    end

    @tag :skip_cuda
    test "Nx.variance/2 with ddof" do
      assert_equal(Nx.variance(Nx.tensor([[1, 2], [3, 4]]), ddof: 1), Nx.tensor(1.6666666))
    end

    test "Nx.standard_deviation/2" do
      t = Nx.tensor([[1, 2], [10, 20]])

      assert_equal(Nx.standard_deviation(t, axes: [0]), Nx.tensor([4.5, 9.0]))
      assert_equal(Nx.standard_deviation(t, axes: [1]), Nx.tensor([0.5, 5.0]))

      assert_equal(
        Nx.standard_deviation(t, axes: [1], ddof: 1),
        Nx.tensor([0.70710677, 7.071068])
      )

      assert_equal(Nx.standard_deviation(t, keep_axes: true), Nx.tensor([[7.628073]]))

      assert_equal(
        Nx.standard_deviation(Nx.tensor([[1, 2], [0, 4]]) |> Nx.vectorize(:x)),
        Nx.tensor([0.5, 2.0]) |> Nx.vectorize(:x)
      )
    end

    @tag :skip_cuda
    test "Nx.standard_deviation/2 with float rounding" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert_equal(Nx.standard_deviation(t), Nx.tensor(1.118034))
      assert_equal(Nx.standard_deviation(t, ddof: 1), Nx.tensor(1.2909944))

      assert_equal(
        Nx.standard_deviation(Nx.tensor([[1, 2], [10, 20]]), axes: [0], ddof: 1),
        Nx.tensor([6.363961, 12.727922])
      )
    end

    test "Nx.reduce/4" do
      t = Nx.tensor([1, 2, 3], names: [:x])
      assert_equal(Nx.reduce(t, 0, [axes: [:x]], fn x, y -> Nx.add(x, y) end), Nx.tensor(6))

      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])

      assert_equal(
        Nx.reduce(t, 0, [axes: [:x]], fn x, y -> Nx.add(x, y) end),
        Nx.tensor([[8, 10, 12], [14, 16, 18]], names: [:y, :z])
      )

      assert_equal(
        Nx.reduce(t, 0, [axes: [:y]], fn x, y -> Nx.add(x, y) end),
        Nx.tensor([[5, 7, 9], [17, 19, 21]], names: [:x, :z])
      )

      assert_equal(
        Nx.reduce(t, 0, [axes: [:x, 2]], fn x, y -> Nx.add(x, y) end),
        Nx.tensor([30, 48], names: [:y])
      )

      assert_equal(
        Nx.reduce(t, 0, [axes: [-1]], fn x, y -> Nx.add(x, y) end),
        Nx.tensor([[6, 15], [24, 33]], names: [:x, :y])
      )

      assert_equal(
        Nx.reduce(t, 0, [axes: [:x], keep_axes: true], fn x, y -> Nx.add(x, y) end),
        Nx.tensor([[[8, 10, 12], [14, 16, 18]]], names: [:x, :y, :z])
      )
    end

    # The init value must be an identity of the reduction. Here 10 is not an
    # identity of add, and XLA applies init values at implementation-defined
    # positions, so CUDA adds it more than once. See the XLA operation
    # semantics for Reduce and the StableHLO spec for reduce. This is tracked
    # as a separate issue.
    @tag :skip_cuda
    test "Nx.reduce/4 with a non-identity init value" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]) |> Nx.vectorize(:x)

      assert_equal(
        Nx.reduce(t, 10, [axes: [1]], &Nx.add/2),
        Nx.tensor([[16, 25], [70, 160]]) |> Nx.vectorize(:x)
      )
    end
  end

  test "Nx.to_binary/1" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.Backend)
    assert Nx.to_binary(t) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
    assert Nx.to_binary(t, limit: 2) == <<1::32-native, 2::32-native>>
    assert Nx.to_binary(t, limit: 6) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
  end

  test "Nx.backend_transfer/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, {EXLA.Backend, device_id: 0})
    assert %EXLA.Backend{buffer: %EXLA.DeviceBuffer{}} = et.data

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>

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
    assert Nx.to_binary(nt) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>

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
      assert Nx.to_binary(nt) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>

      nt = Nx.backend_copy(et)
      assert Nx.to_binary(nt) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
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
      assert Nx.to_binary(nt) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>

      nt = Nx.backend_copy(et)
      assert Nx.to_binary(nt) == <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
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
    assert_all_close(t, reconstructed, atol: 1.0e-2, rtol: 1.0e-2)
  end

  test "multi-client" do
    a = Nx.tensor(1, backend: {EXLA.Backend, client: :host})
    b = Nx.tensor(2, backend: {EXLA.Backend, client: :other_host})
    assert_equal(Nx.add(a, b), Nx.tensor(3))
  end

  @tag :multi_device
  test "multi-device" do
    a = Nx.tensor(1, backend: {EXLA.Backend, client: :other_host, device_id: 0})
    assert_equal(Nx.add(a, 2), Nx.tensor(3))

    a = Nx.tensor(1, backend: {EXLA.Backend, client: :other_host, device_id: 1})
    assert_equal(Nx.add(a, 2), Nx.tensor(3))

    a = Nx.tensor([[1]], backend: {EXLA.Backend, client: :other_host, device_id: 0})
    assert %{device_id: 0, client_name: :other_host} = a.data.buffer
    assert %{device_id: 0, client_name: :other_host} = Nx.reshape(a, {1}).data.buffer

    a = Nx.tensor([[1]], backend: {EXLA.Backend, client: :other_host, device_id: 1})
    assert %{device_id: 1, client_name: :other_host} = a.data.buffer
    assert %{device_id: 1, client_name: :other_host} = Nx.reshape(a, {1}).data.buffer
  end

  @tag :multi_device
  test "stack and concatenate should end up in the same client" do
    t_0 =
      Nx.tensor([1], backend: {EXLA.Backend, client: :no_automatic_transfers_host, device_id: 0})

    t_1 =
      Nx.tensor([1], backend: {EXLA.Backend, client: :no_automatic_transfers_host, device_id: 1})

    t_stack_0 = Nx.stack([t_0, t_1])
    t_concat_0 = Nx.concatenate([t_0, t_1])

    assert t_stack_0.data.buffer.client_name == :no_automatic_transfers_host
    assert t_stack_0.data.buffer.device_id == 1

    assert t_concat_0.data.buffer.client_name == :no_automatic_transfers_host
    assert t_concat_0.data.buffer.device_id == 1

    t_stack_1 = Nx.stack([t_1, t_0])
    t_concat_1 = Nx.concatenate([t_1, t_0])

    assert t_stack_1.data.buffer.client_name == :no_automatic_transfers_host
    assert t_stack_1.data.buffer.device_id == 0

    assert t_concat_1.data.buffer.client_name == :no_automatic_transfers_host
    assert t_concat_1.data.buffer.device_id == 0
  end

  test "Kernel.inspect/2" do
    t = Nx.tensor([1, 2, 3, 4], backend: EXLA.Backend)

    assert inspect(t) ==
             """
             #Nx.Tensor<
               s32[4]
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

    test "raises on invalid client" do
      assert_raise ArgumentError,
                   ~r"could not find EXLA client named :unknown",
                   fn ->
                     Nx.backend_transfer(Nx.tensor([1, 2]), {EXLA.Backend, client: :unknown})
                   end
    end
  end

  describe "access" do
    test "multiple indexes" do
      tensor = Nx.eye({4, 4})
      index = Nx.u32(2)
      swap = Nx.s64(0)
      assert tensor[[index, swap]] |> Nx.to_number() == 0
      assert tensor[[0, swap]] |> Nx.to_number() == 1
    end
  end

  test "conjugate" do
    assert inspect(Nx.conjugate(~VEC[1 2-0i 3+0i 0-i 0-2i])) =~
             "1.0-0.0i, 2.0+0.0i, 3.0-0.0i, 0.0+1.0i, 0.0+2.0i"
  end

  test "gather vectorized regression" do
    gradients =
      Nx.tensor(
        [
          [1.0, 1.0],
          [-1.0, 1.0],
          [1.0, -1.0],
          [-1.0, -1.0]
        ],
        backend: EXLA.Backend
      )

    i =
      Nx.tensor([[0, 2, 3, 2, 2, 2, 2, 1]], type: {:u, 16}, backend: EXLA.Backend)
      |> Nx.vectorize([:x, :octaves])

    result = Nx.gather(gradients, Nx.reshape(i, {1}))

    assert_equal(
      result,
      Nx.tensor([
        [
          [1.0, 1.0],
          [1.0, -1.0],
          [-1.0, -1.0],
          [1.0, -1.0],
          [1.0, -1.0],
          [1.0, -1.0],
          [1.0, -1.0],
          [-1.0, 1.0]
        ]
      ])
      |> Nx.vectorize([:x, :octaves])
    )
  end

  describe "quantized types" do
    test "s2" do
      tensor = Nx.s2(-1)
      assert <<-1::2-signed-native>> = Nx.to_binary(tensor)

      tensor = Nx.s2([-2, -1, 1])
      assert tensor.type == {:s, 2}

      assert <<-2::2-signed-native, -1::2-signed-native, 1::2-signed-native>> =
               Nx.to_binary(tensor)

      assert [-2, -1, 1] = Nx.to_flat_list(tensor)
      assert 0 = Nx.byte_size(tensor)
      assert 6 = Nx.bit_size(tensor)

      tensor = Nx.s2([-2, -1, 0, 1, 0, -1, -2])
      assert 1 = Nx.byte_size(tensor)
      assert 14 = Nx.bit_size(tensor)
    end

    test "s4" do
      tensor = Nx.s4(-1)
      assert <<-1::4-signed-native>> = Nx.to_binary(tensor)

      tensor = Nx.s4([-8, -1, 7])
      assert tensor.type == {:s, 4}

      assert <<-8::4-signed-native, -1::4-signed-native, 7::4-signed-native>> =
               Nx.to_binary(tensor)

      assert [-8, -1, 7] = Nx.to_flat_list(tensor)
      assert 1 = Nx.byte_size(tensor)
      assert 12 = Nx.bit_size(tensor)

      tensor = Nx.s4([-8, -3, 0, 7, 0, -3, -8])
      assert 3 = Nx.byte_size(tensor)
      assert 28 = Nx.bit_size(tensor)
    end

    test "u2" do
      tensor = Nx.u2(1)
      assert <<1::2-native>> = Nx.to_binary(tensor)

      tensor = Nx.u2([1, 2, 3])
      assert tensor.type == {:u, 2}
      assert <<1::2-native, 2::2-native, 3::2-native>> = Nx.to_binary(tensor)
      assert [1, 2, 3] = Nx.to_flat_list(tensor)
      assert 0 = Nx.byte_size(tensor)
      assert 6 = Nx.bit_size(tensor)

      tensor = Nx.u2([0, 1, 2, 3, 2, 1, 0])
      assert 1 = Nx.byte_size(tensor)
      assert 14 = Nx.bit_size(tensor)
    end

    test "u4" do
      tensor = Nx.u4(1)
      assert <<1::4-native>> = Nx.to_binary(tensor)

      tensor = Nx.u4([0, 7, 15])
      assert tensor.type == {:u, 4}
      assert <<0::4-native, 7::4-native, 15::4-native>> = Nx.to_binary(tensor)
      assert [0, 7, 15] = Nx.to_flat_list(tensor)
      assert 1 = Nx.byte_size(tensor)
      assert 12 = Nx.bit_size(tensor)

      tensor = Nx.u4([0, 1, 2, 3, 13, 14, 15])
      assert 3 = Nx.byte_size(tensor)
      assert 28 = Nx.bit_size(tensor)
    end
  end
end
