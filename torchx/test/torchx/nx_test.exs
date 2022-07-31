defmodule Torchx.NxTest do
  use Torchx.Case, async: true

  import Nx, only: :sigils

  alias Torchx.Backend, as: TB
  doctest TB

  @types [{:s, 8}, {:u, 8}, {:s, 16}, {:s, 32}, {:s, 64}, {:bf, 16}, {:f, 32}, {:f, 64}]
  @bf16_and_ints [{:s, 8}, {:u, 8}, {:s, 16}, {:s, 32}, {:s, 64}, {:bf, 16}]
  @ints [{:s, 8}, {:u, 8}, {:s, 16}, {:s, 32}, {:s, 64}]
  @ops [:add, :subtract, :divide, :remainder, :multiply, :power, :atan2, :min, :max]
  @ops_unimplemented_for_bfloat [:remainder, :atan2, :power]
  @ops_with_bfloat_specific_result [:divide]
  @bitwise_ops [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift]
  @logical_ops [
    :equal,
    :not_equal,
    :greater,
    :less,
    :greater_equal,
    :less_equal,
    :logical_and,
    :logical_or,
    :logical_xor
  ]
  @unary_ops [:abs, :bitwise_not, :ceil, :floor, :negate, :round, :sign]

  defp test_binary_op(op, data_a \\ [[5, 6], [7, 8]], data_b \\ [[1, 2], [3, 4]], type_a, type_b) do
    a = Nx.tensor(data_a, type: type_a)
    b = Nx.tensor(data_b, type: type_b)
    c = Kernel.apply(Nx, op, [a, b])

    binary_a = Nx.backend_copy(a, Nx.BinaryBackend)
    binary_b = Nx.backend_transfer(b, Nx.BinaryBackend)
    binary_c = Kernel.apply(Nx, op, [binary_a, binary_b])
    assert_equal(c, binary_c)

    mixed_c = Kernel.apply(Nx, op, [a, binary_b])
    assert_equal(mixed_c, binary_c)
  end

  defp test_unary_op(op, data \\ [[1, 2], [3, 4]], type) do
    t = Nx.tensor(data, type: type)
    r = Kernel.apply(Nx, op, [t])

    binary_t = Nx.backend_copy(t, Nx.BinaryBackend)
    binary_r = Kernel.apply(Nx, op, [binary_t])
    assert_equal(r, binary_r)
  end

  describe "binary ops" do
    for op <- @ops ++ @logical_ops,
        type_a <- @types,
        type_b <- @types,
        not (op in (@ops_unimplemented_for_bfloat ++ @ops_with_bfloat_specific_result) and
               Nx.Type.merge(type_a, type_b) == {:bf, 16}) do
      test "#{op}(#{Nx.Type.to_string(type_a)}, #{Nx.Type.to_string(type_b)})" do
        op = unquote(op)
        type_a = unquote(type_a)
        type_b = unquote(type_b)

        test_binary_op(op, type_a, type_b)
      end
    end
  end

  # quotient/2 works only with integers, so we put it here.
  describe "binary bitwise ops" do
    for op <- @bitwise_ops ++ [:quotient],
        type_a <- @ints,
        type_b <- @ints do
      test "#{op}(#{Nx.Type.to_string(type_a)}, #{Nx.Type.to_string(type_b)})" do
        op = unquote(op)
        type_a = unquote(type_a)
        type_b = unquote(type_b)

        test_binary_op(op, type_a, type_b)
      end
    end
  end

  describe "unary ops" do
    for op <- @unary_ops -- [:bitwise_not],
        type <- @types do
      test "#{op}(#{Nx.Type.to_string(type)})" do
        test_unary_op(unquote(op), unquote(type))
      end
    end

    for type <- @ints do
      test "bitwise_not(#{Nx.Type.to_string(type)})" do
        test_unary_op(:bitwise_not, unquote(type))
      end
    end

    test "all does not fail when more than one axis is passed" do
      assert_all_close(
        Nx.all(
          Nx.tensor([[[[1]]]]),
          axes: [0, 1]
        ),
        Nx.tensor([[1]])
      )
    end

    test "fft" do
      assert_all_close(
        Nx.fft(Nx.tensor([1, 1, 0, 0]), length: 5),
        ~V[2.0+0.0i 1.3090-0.9511i 0.1909-0.5877i 0.1909+0.5877i 1.3090+0.9510i]
      )

      assert_all_close(
        Nx.fft(Nx.tensor([1, 1, 0, 0, 2, 3]), length: 4),
        ~V[2.0+0.0i 1.0-1.0i 0.0+0.0i 1.0+1.0i]
      )

      assert_all_close(
        Nx.fft(Nx.tensor([1, 1, 0]), length: :power_of_two),
        ~V[2.0+0.0i 1.0-1.0i 0.0+0.0i 1.0+1.0i]
      )
    end

    test "fft - n dim tensor" do
      assert_all_close(
        Nx.fft(
          Nx.tensor([
            [
              [1, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0]
            ],
            [
              [0, 1, 0, 0],
              [1, 0, 0, 0],
              [1, 1, 0, 0]
            ]
          ]),
          length: :power_of_two
        ),
        Nx.stack([
          ~M[
                2 1.0-1.0i 0 1.0+1.0i
                1 1 1 1
                1 -1i -1 1i
              ],
          ~M[
                1 -1i -1 1i
                1 1 1 1
                2 1.0-1.0i 0 1.0+1.0i
              ]
        ])
      )

      assert_all_close(
        Nx.fft(
          Nx.tensor([
            [
              [1, 1, 0, 0, 1, 2],
              [1, 0, 0, 0, 3, 4],
              [0, 1, 0, 0, 5, 6]
            ],
            [
              [0, 1, 0, 0, 7, 8],
              [1, 0, 0, 0, 9, 10],
              [1, 1, 0, 0, 11, 12]
            ]
          ]),
          length: 4
        ),
        Nx.stack([
          ~M[
                2 1.0-1.0i 0 1.0+1.0i
                1 1 1 1
                1 -1i -1 1i
              ],
          ~M[
                1 -1i -1 1i
                1 1 1 1
                2 1.0-1.0i 0 1.0+1.0i
              ]
        ])
      )

      assert_all_close(
        Nx.fft(
          Nx.tensor([
            [
              [1, 1, 0],
              [1, 0, 0],
              [0, 1, 0]
            ],
            [
              [0, 1, 0],
              [1, 0, 0],
              [1, 1, 0]
            ]
          ]),
          length: 4
        ),
        Nx.stack([
          ~M[
                2 1.0-1.0i 0 1.0+1.0i
                1 1 1 1
                1 -1i -1 1i
              ],
          ~M[
                1 -1i -1 1i
                1 1 1 1
                2 1.0-1.0i 0 1.0+1.0i
              ]
        ])
      )
    end

    test "ifft" do
      assert_all_close(
        Nx.ifft(~V[5 5 5 5 5],
          length: 5
        ),
        Nx.tensor([5, 0, 0, 0, 0])
      )

      assert_all_close(
        Nx.ifft(~V[2.0+0.0i 1.0-1.0i 0.0+0.0i 1.0+1.0i 5 6], length: 4),
        Nx.tensor([1, 1, 0, 0])
      )

      assert_all_close(
        Nx.ifft(~V[2 0 0], length: :power_of_two),
        Nx.tensor([0.5, 0.5, 0.5, 0.5])
      )
    end

    test "ifft - n dim tensor" do
      assert_all_close(
        Nx.ifft(
          Nx.stack([
            ~M[
                2 1.0-1.0i 0 1.0+1.0i
                1 1 1 1
                1 -1i -1 1i
              ],
            ~M[
                1 -1i -1 1i
                1 1 1 1
                2 1.0-1.0i 0 1.0+1.0i
              ]
          ]),
          length: :power_of_two
        ),
        Nx.tensor([
          [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
          ],
          [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0]
          ]
        ])
      )

      assert_all_close(
        Nx.ifft(
          Nx.tensor([
            [
              [4, 4, 0, 0, 1, 2],
              [4, 0, 0, 0, 3, 4],
              [0, 4, 0, 0, 5, 6]
            ],
            [
              [0, 4, 0, 0, 7, 8],
              [4, 0, 0, 0, 9, 10],
              [4, 4, 0, 0, 11, 12]
            ]
          ]),
          length: 4
        ),
        Nx.stack([
          ~M[
                2 1.0+1.0i 0 1.0-1.0i
                1 1 1 1
                1 1i -1 -1i
              ],
          ~M[
                1 1i -1 -1i
                1 1 1 1
                2 1.0+1.0i 0 1.0-1.0i
              ]
        ])
      )

      assert_all_close(
        Nx.ifft(
          Nx.tensor([
            [
              [4, 4, 0],
              [4, 0, 0],
              [0, 4, 0]
            ],
            [
              [0, 4, 0],
              [4, 0, 0],
              [4, 4, 0]
            ]
          ]),
          length: 4
        ),
        Nx.stack([
          ~M[
                2 1.0+1.0i 0 1.0-1.0i
                1 1 1 1
                1 1i -1 -1i
              ],
          ~M[
                1 1i -1 -1i
                1 1 1 1
                2 1.0+1.0i 0 1.0-1.0i
              ]
        ])
      )
    end
  end

  # Division and power with bfloat16 are special cases in PyTorch,
  # because it upcasts bfloat16 args to float for numerical accuracy purposes.
  # So, e.g., the result of division is different from what direct bf16 by bf16 division gives us.
  # I.e. 1/5 == 0.19921875 in direct bf16 division and 0.2001953125 when dividing floats
  # converting them to bf16 afterwards (PyTorch style).
  describe "bfloat16" do
    for type_a <- @bf16_and_ints,
        type_b <- @bf16_and_ints,
        type_a == {:bf, 16} or type_b == {:bf, 16} do
      test "divide(#{Nx.Type.to_string(type_a)}, #{Nx.Type.to_string(type_b)})" do
        type_a = unquote(type_a)
        type_b = unquote(type_b)

        a = Nx.tensor([[1, 2], [3, 4]], type: type_a)
        b = Nx.tensor([[5, 6], [7, 8]], type: type_b)

        c = Nx.divide(a, b)

        assert_equal(
          c,
          Nx.tensor([[0.2001953125, 0.333984375], [0.427734375, 0.5]])
        )
      end
    end
  end

  describe "creation" do
    test "eye" do
      t = Nx.eye({9, 9}) |> Nx.backend_transfer()
      one = Nx.tensor(1, backend: Nx.BinaryBackend)
      zero = Nx.tensor(0, backend: Nx.BinaryBackend)

      for i <- 0..8, j <- 0..8 do
        assert (i == j and t[i][j] == one) or t[i][j] == zero
      end
    end

    test "iota" do
      t = Nx.iota({2, 3})

      assert_equal(t, Nx.tensor([[0, 1, 2], [3, 4, 5]]))
    end

    test "dot with vectors" do
      t1 = Nx.tensor([1, 2, 3])
      t2 = Nx.tensor([4, 5, 6])

      out = Nx.dot(t1, t2)

      assert_equal(out, Nx.tensor(32))
    end

    test "dot with matrices" do
      t1 = Nx.tensor([[1, 2], [3, 4]])
      t2 = Nx.tensor([[5, 6], [7, 8]])

      out = Nx.dot(t1, t2)

      assert_equal(out, Nx.tensor([[19, 22], [43, 50]]))
    end

    test "dot with vector and scalar" do
      t = Nx.tensor([[1, 2, 3]])
      assert Nx.shape(t) == {1, 3}
      out = Nx.dot(t, 3)

      assert Nx.shape(out) == {1, 3}
      assert_equal(out, Nx.tensor([[3, 6, 9]]))
    end

    test "dot does not re-sort the contracting axes" do
      t1 = Nx.iota({2, 7, 8, 3, 1})
      t2 = Nx.iota({1, 8, 3, 7, 3})

      out = Nx.dot(t1, [3, 1, 2], t2, [2, 3, 1])

      assert {2, 1, 1, 3} == out.shape

      assert_equal(
        out,
        Nx.tensor([
          [
            [
              [3_731_448, 3_745_476, 3_759_504]
            ]
          ],
          [
            [
              [10_801_560, 10_843_812, 10_886_064]
            ]
          ]
        ])
      )
    end

    test "dot currently raises when using batching" do
      # Batching not supported for now. Once it is
      # supported doctests for dot should be restablished

      t1 = Nx.iota({3, 2, 4, 1})
      t2 = Nx.iota({3, 4, 2, 2})

      assert_raise(FunctionClauseError, fn ->
        Nx.dot(t1, [1, 2], [0], t2, [2, 1], [0])
      end)
    end

    test "make_diagonal" do
      t =
        [1, 2, 3]
        |> Nx.tensor()
        |> Nx.make_diagonal()

      assert_equal(t, Nx.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))
    end

    test "random_uniform" do
      t = Nx.random_uniform({30, 50})

      t
      |> Nx.backend_transfer()
      |> Nx.to_flat_list()
      |> Enum.all?(&(&1 > 0.0 and &1 < 1.0))
    end

    test "random_uniform with range" do
      t = Nx.random_uniform({30, 50}, 7, 12)

      t
      |> Nx.backend_transfer()
      |> Nx.to_flat_list()
      |> Enum.all?(&(&1 > 7.0 and &1 < 12.0))
    end

    test "random_normal" do
      t = Nx.random_normal({30, 50})

      t
      |> Nx.backend_transfer()
      |> Nx.to_flat_list()
      |> Enum.all?(&(&1 > 0.0 and &1 < 1.0))
    end

    test "random_normal with range" do
      t = Nx.random_normal({30, 50}, 7.0, 3.0)

      t
      |> Nx.backend_transfer()
      |> Nx.to_flat_list()
      |> Enum.all?(&(&1 > 7.0 - 3.0 and &1 < 7.0 + 3.0))
    end
  end

  describe "rounding error tests" do
    test "atanh/1" do
      assert_all_close(Nx.tensor(0.5493), Nx.atanh(Nx.tensor(0.5)))
    end

    test "ceil/1" do
      assert_all_close(Nx.tensor(-0.0), Nx.ceil(Nx.tensor(-0.5)))
      assert_all_close(Nx.tensor(1.0), Nx.ceil(Nx.tensor(0.5)))
    end

    test "cos/1" do
      assert_all_close(
        Nx.tensor([-1.0, 0.4999, -1.0]),
        Nx.cos(Nx.tensor([-:math.pi(), :math.pi() / 3, :math.pi()]))
      )
    end

    test "cosh/1" do
      assert_all_close(
        Nx.tensor([11.5919, 1.6002, 11.5919]),
        Nx.cosh(Nx.tensor([-:math.pi(), :math.pi() / 3, :math.pi()]))
      )
    end

    test "erfc/1" do
      assert_all_close(
        Nx.tensor([1.0, 0.4795, 0.0]),
        Nx.erfc(Nx.tensor([0, 0.5, 10_000]))
      )
    end

    test "erf_inv/1" do
      assert_all_close(
        Nx.tensor([0.0, 0.4769, 0.8134]),
        Nx.erf_inv(Nx.tensor([0, 0.5, 0.75]))
      )
    end

    test "round/1" do
      assert_all_close(
        Nx.tensor([-2.0, -0.0, 0.0, 2.0]),
        Nx.round(Nx.tensor([-1.5, -0.5, 0.5, 1.5]))
      )
    end

    test "sigmoid/1" do
      assert_all_close(
        Nx.tensor([0.1824, 0.6224]),
        Nx.sigmoid(Nx.tensor([-1.5, 0.5]))
      )
    end
  end

  describe "Nx.as_type" do
    test "basic conversions" do
      assert Nx.as_type(Nx.tensor([0, 1, 2]), {:f, 32}) |> Nx.backend_transfer() ==
               Nx.tensor([0.0, 1.0, 2.0], backend: Nx.BinaryBackend)

      assert Nx.as_type(Nx.tensor([0.0, 1.0, 2.0]), {:s, 64}) |> Nx.backend_transfer() ==
               Nx.tensor([0, 1, 2], backend: Nx.BinaryBackend)

      assert Nx.as_type(Nx.tensor([0.0, 1.0, 2.0]), {:bf, 16}) |> Nx.backend_transfer() ==
               Nx.tensor([0.0, 1.0, 2.0], type: {:bf, 16}, backend: Nx.BinaryBackend)
    end

    test "non-finite to integer conversions" do
      non_finite =
        Nx.stack([Nx.Constants.infinity(), Nx.Constants.nan(), Nx.Constants.neg_infinity()])

      assert Nx.as_type(non_finite, {:u, 8}) |> Nx.backend_transfer() ==
               Nx.tensor([0, 0, 0], type: {:u, 8}, backend: Nx.BinaryBackend)

      assert Nx.as_type(non_finite, {:s, 16}) |> Nx.backend_transfer() ==
               Nx.tensor([0, 0, 0], type: {:s, 16}, backend: Nx.BinaryBackend)

      assert Nx.as_type(non_finite, {:s, 32}) |> Nx.backend_transfer() ==
               Nx.tensor([-2_147_483_648, -2_147_483_648, -2_147_483_648],
                 type: {:s, 32},
                 backend: Nx.BinaryBackend
               )
    end

    test "non-finite to between floats conversions" do
      non_finite = Nx.stack([Nx.Constants.infinity(), Nx.Constants.neg_infinity()])
      non_finite_binary_backend = Nx.backend_copy(non_finite)

      assert Nx.as_type(non_finite, {:f, 16}) |> Nx.backend_transfer() ==
               Nx.as_type(non_finite_binary_backend, {:f, 16})

      assert Nx.as_type(non_finite, {:f, 64}) |> Nx.backend_transfer() ==
               Nx.as_type(non_finite_binary_backend, {:f, 64})
    end
  end

  describe "Nx.product" do
    test "full tensor products" do
      Nx.tensor(42)
      |> Nx.product()
      |> assert_all_close(Nx.tensor(42))

      Nx.tensor([1, 2, 3], names: [:x])
      |> Nx.product()
      |> assert_all_close(Nx.tensor(6))

      Nx.tensor([[1.0, 2.0], [3.0, 4.0]], names: [:x, :y])
      |> Nx.product()
      |> assert_all_close(Nx.tensor(24))

      Nx.tensor([[1.0, 2.0], [3.0, 4.0]], names: [:x, :y])
      |> Nx.product(axes: [:x, :y])
      |> assert_all_close(Nx.tensor(24))
    end

    test "aggregating over single axis" do
      Nx.tensor([1, 2, 3], names: [:x])
      |> Nx.product(axes: [0])
      |> assert_all_close(Nx.tensor(6))

      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [:x])
      |> assert_all_close(
        Nx.tensor([
          [7, 16, 27],
          [40, 55, 72]
        ])
      )

      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [:y])
      |> assert_all_close(
        Nx.tensor([
          [4, 10, 18],
          [70, 88, 108]
        ])
      )

      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [:x, :z])
      |> assert_all_close(Nx.tensor([3024, 158_400]))

      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [:z])
      |> assert_all_close(
        Nx.tensor([
          [6, 120],
          [504, 1320]
        ])
      )

      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [-3])
      |> assert_all_close(
        Nx.tensor([
          [7, 16, 27],
          [40, 55, 72]
        ])
      )
    end

    test "keep_axes: true" do
      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [:z], keep_axes: true)
      |> assert_all_close(
        Nx.tensor([
          [
            [6],
            [120]
          ],
          [
            [504],
            [1320]
          ]
        ])
      )

      Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      |> Nx.product(axes: [:x, :y, :z], keep_axes: true)
      |> assert_all_close(
        Nx.tensor([
          [
            [479_001_600]
          ]
        ])
      )
    end

    test "validates axis" do
      assert_raise ArgumentError, "given axis (2) invalid for shape with rank 2", fn ->
        Nx.product(Nx.tensor([[1, 2]]), axes: [2])
      end
    end
  end

  describe "pad" do
    test "2d" do
      Nx.tensor(1)
      |> Nx.pad(0, [])
      |> assert_all_close(Nx.tensor(1))

      Nx.tensor([1, 2, 3], names: [:data])
      |> Nx.pad(0, [{1, 1, 0}])
      |> assert_all_close(Nx.tensor([0, 1, 2, 3, 0]))

      Nx.tensor([[1, 2, 3], [4, 5, 6]])
      |> Nx.pad(0, [{1, 1, 0}, {1, 1, 0}])
      |> assert_all_close(
        Nx.tensor([
          [0, 0, 0, 0, 0],
          [0, 1, 2, 3, 0],
          [0, 4, 5, 6, 0],
          [0, 0, 0, 0, 0]
        ])
      )
    end

    test "3d" do
      Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      |> Nx.pad(0, [{0, 2, 0}, {1, 1, 0}, {1, 0, 0}])
      |> assert_all_close(
        Nx.tensor([
          [
            [0, 0, 0],
            [0, 1, 2],
            [0, 3, 4],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 5, 6],
            [0, 7, 8],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ]
        ])
      )

      Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      |> Nx.pad(0, [{1, 0, 0}, {1, 1, 0}, {0, 1, 0}])
      |> assert_all_close(
        Nx.tensor([
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [5, 6, 0],
            [7, 8, 0],
            [0, 0, 0]
          ]
        ])
      )

      Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
      |> Nx.pad(0.0, [{1, 2, 0}, {1, 0, 0}, {0, 1, 0}])
      |> assert_all_close(
        Nx.tensor([
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [5.0, 6.0, 0.0],
            [7.0, 8.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ]
        ])
      )
    end
  end

  describe "conv" do
    test "test input_permutation" do
      left = Nx.iota({9})
      left = Nx.reshape(left, {1, 1, 3, 3})
      right = Nx.iota({8})
      right = Nx.reshape(right, {4, 1, 2, 1})

      Nx.conv(left, right,
        strides: 2,
        padding: :same,
        kernel_dilation: [2, 1],
        input_permutation: [3, 1, 2, 0]
      )
      |> assert_all_close(
        Nx.tensor([
          [
            [
              [3.0],
              [0.0]
            ],
            [
              [9.0],
              [6.0]
            ],
            [
              [15.0],
              [12.0]
            ],
            [
              [21.0],
              [18.0]
            ]
          ],
          [
            [
              [4.0],
              [0.0]
            ],
            [
              [12.0],
              [8.0]
            ],
            [
              [20.0],
              [16.0]
            ],
            [
              [28.0],
              [24.0]
            ]
          ],
          [
            [
              [5.0],
              [0.0]
            ],
            [
              [15.0],
              [10.0]
            ],
            [
              [25.0],
              [20.0]
            ],
            [
              [35.0],
              [30.0]
            ]
          ]
        ])
      )
    end
  end

  describe "window_max" do
    test "works with default opts" do
      result =
        Nx.window_max(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]), {1, 2, 1})

      assert_all_close(
        result,
        Nx.tensor([
          [
            [4, 5, 6]
          ],
          [
            [4, 5, 6]
          ]
        ])
      )
    end

    test "supports padding" do
      t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      result = Nx.window_max(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

      assert_all_close(
        result,
        Nx.tensor([
          [
            [-3.4028234663852886e38, 4.0, 2.0, 3.0, -3.4028234663852886e38],
            [-3.4028234663852886e38, 2.0, 5.0, 6.5, -3.4028234663852886e38]
          ],
          [
            [
              -3.4028234663852886e38,
              1.2000000476837158,
              2.200000047683716,
              3.200000047683716,
              -3.4028234663852886e38
            ],
            [-3.4028234663852886e38, 4.0, 5.0, 6.199999809265137, -3.4028234663852886e38]
          ]
        ])
      )
    end

    # window dilations temporarily broken
    @tag :skip
    test "works with non-default options" do
      t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      result = Nx.window_max(t, {1, 1, 2}, opts)

      assert_all_close(
        result,
        Nx.tensor([
          [
            [4, 3],
            [4, 7]
          ]
        ])
      )
    end
  end

  describe "window_min" do
    test "works with default opts" do
      result =
        Nx.window_min(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]), {1, 2, 1})

      assert_all_close(
        result,
        Nx.tensor([
          [
            [1, 2, 3]
          ],
          [
            [1, 2, 3]
          ]
        ])
      )
    end

    test "fails with non-zero padding" do
      t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      result = Nx.window_min(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

      assert_all_close(
        result,
        Nx.tensor([
          [
            [3.4028234663852886e38, 4.0, 2.0, 3.0, 3.4028234663852886e38],
            [3.4028234663852886e38, 2.0, 5.0, 6.5, 3.4028234663852886e38]
          ],
          [
            [
              3.4028234663852886e38,
              1.2000000476837158,
              2.200000047683716,
              3.200000047683716,
              3.4028234663852886e38
            ],
            [3.4028234663852886e38, 4.0, 5.0, 6.199999809265137, 3.4028234663852886e38]
          ]
        ])
      )
    end

    # window dilations temporarily broken
    @tag :skip
    test "works with non-default options" do
      t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      result = Nx.window_min(t, {1, 1, 2}, opts)

      assert_all_close(
        result,
        Nx.tensor([
          [
            [1, 2],
            [1, 2]
          ]
        ])
      )
    end
  end

  describe "window_sum" do
    test "works with simple params" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      result = Nx.window_sum(t, {1, 2, 1})

      assert_all_close(
        result,
        Nx.tensor([
          [
            [5, 7, 9]
          ],
          [
            [5, 7, 9]
          ]
        ])
      )
    end

    test "works with non default options" do
      t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      result = Nx.window_sum(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

      assert_all_close(
        result,
        Nx.tensor([
          [
            [0.0, 4.0, 2.0, 3.0, 0.0],
            [0.0, 2.0, 5.0, 6.5, 0.0]
          ],
          [
            [0.0, 1.2000000476837158, 2.200000047683716, 3.200000047683716, 0.0],
            [0.0, 4.0, 5.0, 6.199999809265137, 0.0]
          ]
        ])
      )
    end
  end

  describe "window_product" do
    test "works with simple params" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      result = Nx.window_product(t, {1, 2, 1})

      assert_all_close(
        result,
        Nx.tensor([
          [
            [4, 10, 18]
          ],
          [
            [4, 10, 18]
          ]
        ])
      )
    end

    test "works with non default options" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

      result =
        Nx.window_product(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])

      assert_all_close(
        result,
        Nx.tensor([
          [
            [1, 1],
            [1, 324]
          ],
          [
            [1, 1],
            [1, 18]
          ]
        ])
      )
    end
  end
end
