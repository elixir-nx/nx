defmodule Torchx.BackendTest do
  use Torchx.Case, async: true

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

    binary_a = Nx.backend_transfer(a, Nx.BinaryBackend)
    binary_b = Nx.backend_transfer(b, Nx.BinaryBackend)
    binary_c = Kernel.apply(Nx, op, [binary_a, binary_b])
    assert Nx.backend_transfer(c) == binary_c

    mixed_c = Kernel.apply(Nx, op, [a, binary_b])
    assert Nx.backend_transfer(mixed_c) == binary_c
  end

  defp test_unary_op(op, data \\ [[1, 2], [3, 4]], type) do
    t = Nx.tensor(data, type: type)
    r = Kernel.apply(Nx, op, [t])

    binary_t = Nx.backend_transfer(t, Nx.BinaryBackend)
    binary_r = Kernel.apply(Nx, op, [binary_t])
    assert Nx.backend_transfer(r) == binary_r
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

    test "all? fails with more than one axis passed" do
      assert_raise ArgumentError, ":axes option only accepts a single axis per call", fn ->
        t = Nx.tensor([[[1]]])
        Torchx.Backend.all?(t, t, axes: [0, 1])
      end
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

        assert Nx.backend_transfer(c) ==
                 Nx.tensor([[0.2001953125, 0.333984375], [0.427734375, 0.5]],
                   type: {:bf, 16},
                   backend: Nx.BinaryBackend
                 )
      end
    end
  end

  describe "vectors" do
    for type_a <- @types,
        type_b <- @types do
      test "outer(#{Nx.Type.to_string(type_a)}, #{Nx.Type.to_string(type_b)})" do
        type_a = unquote(type_a)
        type_b = unquote(type_b)

        test_binary_op(:outer, [1, 2, 3, 4], [5, 6, 7, 8], type_a, type_b)
      end
    end
  end

  describe "aggregates" do
    test "sum throws on type mismatch" do
      t = Nx.tensor([[101, 102], [103, 104]], type: {:u, 8})

      assert_raise(
        ArgumentError,
        "Torchx does not support unsigned 64 bit integer (explicitly cast the input tensor to a signed integer before taking sum)",
        fn -> Nx.sum(t) end
      )
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

      assert Nx.backend_transfer(t) ==
               Nx.tensor([[0, 1, 2], [3, 4, 5]], backend: Nx.BinaryBackend)
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

    test "logistic/1" do
      assert_all_close(
        Nx.tensor([0.1824, 0.6224]),
        Nx.logistic(Nx.tensor([-1.5, 0.5]))
      )
    end
  end

  describe "Nx.LinAlg.triangular_solve" do
    test "base case 1D (s64)" do
      a = Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])

      result = Nx.LinAlg.triangular_solve(a, Nx.tensor([4, 2, 4, 2]))

      assert_all_close(result, Nx.tensor([1.33333337, -0.6666666, 2.6666667, -1.33333]))
    end

    test "base case 1D (f64)" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      %{type: {:f, 64}} = result = Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]))
      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "base case 2D" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1, 2, 3], [2, 2, 4], [2, 0, 1]])
      result = Nx.LinAlg.triangular_solve(a, b)

      expected =
        Nx.tensor([
          [1.0, 2.0, 3.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0]
        ])

      assert_all_close(result, expected)
    end

    test "lower: false" do
      a = Nx.tensor([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 3]])
      b = Nx.tensor([2, 4, 2, 4])
      result = Nx.LinAlg.triangular_solve(a, b, lower: false)

      expected =
        Nx.tensor([
          -1.3333,
          2.66666,
          -0.6666,
          1.33333
        ])

      assert_all_close(result, expected)
    end

    test "left_side: false" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      b = Nx.tensor([[0, 2, 1], [1, 1, 0], [3, 3, 1]])

      assert_raise ArgumentError, "left_side: false option not supported in Torchx", fn ->
        Nx.LinAlg.triangular_solve(a, b, left_side: false)
      end
    end

    test "transform_a: :transpose" do
      a = Nx.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], type: {:f, 64})
      b = Nx.tensor([1, 2, 1])
      result = Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose, lower: false)

      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "explicit transform_a: :none" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      b = Nx.tensor([1, 2, 1])
      result = Nx.LinAlg.triangular_solve(a, b, transform_a: :none)

      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "explicit left_side: true" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      b = Nx.tensor([[0, 2], [3, 0], [0, 0]])
      result = Nx.LinAlg.triangular_solve(a, b, left_side: true)

      assert_all_close(
        result,
        Nx.tensor([
          [0.0, 2.0],
          [3.0, -2.0],
          [-6.0, 2.0]
        ])
      )
    end

    test "invalid a shape" do
      assert_raise ArgumentError, "expected a square matrix, got matrix with shape: {2, 4}", fn ->
        Nx.LinAlg.triangular_solve(
          Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0]]),
          Nx.tensor([4, 2, 4, 2])
        )
      end
    end

    test "incompatible dims" do
      assert_raise ArgumentError, "incompatible dimensions for a and b on triangular solve", fn ->
        Nx.LinAlg.triangular_solve(
          Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
          Nx.tensor([4])
        )
      end
    end

    test "singular matrix" do
      assert_raise ArgumentError, "can't solve for singular matrix", fn ->
        Nx.LinAlg.triangular_solve(
          Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),
          Nx.tensor([4, 2, 4, 2])
        )
      end
    end

    test "complex numbers not supported" do
      assert_raise ArgumentError, "complex numbers not supported yet", fn ->
        a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
        Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :conjugate)
      end
    end

    test "validates transform_a" do
      assert_raise ArgumentError,
                   "invalid value for :transform_a option, expected :none, :transpose, or :conjugate, got: :other",
                   fn ->
                     a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
                     Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :other)
                   end
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

  describe "Nx.LinAlg.lu" do
    test "property" do
      for _ <- 1..20 do
        a = Nx.random_uniform({3, 3})
        {p, l, u} = Nx.LinAlg.lu(a)

        a_reconstructed = p |> Nx.dot(l) |> Nx.dot(u)

        assert_all_close(a, a_reconstructed)
      end
    end

    test "invalid a shape" do
      assert_raise ArgumentError,
                   "tensor must have as many rows as columns, got shape: {3, 4}",
                   fn ->
                     Nx.LinAlg.lu(Nx.tensor([[1, 1, 1, 1], [-1, 4, 4, -1], [4, -2, 2, 0]]))
                   end
    end
  end
end
