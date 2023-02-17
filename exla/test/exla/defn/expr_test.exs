defmodule EXLA.Defn.ExprTest do
  use EXLA.Case, async: true

  import Nx, only: :sigils
  import Nx.Defn

  defp evaluate(fun, args) do
    fun |> Nx.Defn.jit(compiler: Nx.Defn.Evaluator) |> apply(args)
  end

  describe "tuples" do
    defn add_subtract_tuple(a, b), do: {a + b, a - b}

    test "on results" do
      assert_equal(add_subtract_tuple(2, 3), {Nx.tensor(5), Nx.tensor(-1)})

      assert_equal(
        add_subtract_tuple(Nx.tensor([-1, 0, 1]), 10),
        {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
      )
    end

    defn pattern_tuple({a, b}), do: a + b

    test "on patterns" do
      assert_equal(pattern_tuple({2, 3}), Nx.tensor(5))

      assert_equal(
        pattern_tuple({Nx.tensor([1, 2]), Nx.tensor([[3], [4]])}),
        Nx.tensor([[4, 5], [5, 6]])
      )
    end

    defn calls_pattern_tuple(a, b), do: pattern_tuple({a, b})

    test "on inlined tuples" do
      assert_equal(calls_pattern_tuple(2, 3), Nx.tensor(5))

      assert_equal(
        calls_pattern_tuple(Nx.tensor([1, 2]), Nx.tensor([[3], [4]])),
        Nx.tensor([[4, 5], [5, 6]])
      )
    end
  end

  describe "tensor constants" do
    @two 2
    defn constants, do: @two
    defn add_two_attribute(t), do: t + @two

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn add_2x2_attribute(t), do: t + @two_per_two

    test "handles tensors as constants" do
      assert_equal(constants(), Nx.tensor(2))
    end

    test "expands module attributes to scalars" do
      assert_equal(add_two_attribute(1), Nx.tensor(3))
      assert_equal(add_two_attribute(Nx.tensor([1, 2, 3])), Nx.tensor([3, 4, 5]))
    end

    test "expands module attributes to tensors" do
      assert_equal(add_2x2_attribute(1), Nx.tensor([[2, 3], [4, 5]]))
      assert_equal(add_2x2_attribute(Nx.tensor([1, 2])), Nx.tensor([[2, 4], [4, 6]]))
    end
  end

  describe "non finite" do
    defn infinity, do: Nx.Constants.infinity()
    defn neg_infinity, do: Nx.Constants.neg_infinity()
    defn nan, do: Nx.Constants.nan()

    test "handles non-finite constants correctly" do
      assert_equal(infinity(), Nx.Constants.infinity())
      assert_equal(neg_infinity(), Nx.Constants.neg_infinity())
      assert_equal(nan(), Nx.Constants.nan())
    end

    defn negate_infinity, do: Nx.negate(Nx.Constants.infinity())
    defn negate_neg_infinity, do: Nx.negate(Nx.Constants.infinity())

    test "sanity check constants" do
      assert_equal(negate_infinity(), Nx.Constants.neg_infinity())
      assert_equal(infinity(), Nx.Constants.infinity())
    end
  end

  describe "float16" do
    defn return_float, do: Nx.tensor(1, type: {:f, 16})

    test "supports float16 return types" do
      assert_equal(return_float(), Nx.tensor(1, type: {:f, 16}))
    end
  end

  describe "complex" do
    defn return_complex, do: Nx.complex(1, 2)
    defn return_complex_tensor, do: Nx.broadcast(Nx.complex(1, 2), {3, 3, 3})

    test "supports complex return types" do
      assert_equal(return_complex(), Nx.tensor(Complex.new(1, 2)))
      assert_equal(return_complex_tensor(), Nx.broadcast(Complex.new(1, 2), {3, 3, 3}))
    end
  end

  describe "conjugate" do
    defn conjugate(x), do: Nx.conjugate(x)

    test "correctly returns complex conjugate" do
      assert_equal(conjugate(Nx.tensor(Complex.new(1, 2))), Nx.tensor(Complex.new(1, -2)))
      # This differs from the Nx doctest, which I believe should also return -0
      assert_equal(conjugate(Nx.tensor(1)), Nx.tensor(Complex.new(1, -0.0)))

      assert_equal(
        conjugate(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([Complex.new(1, -2), Complex.new(2, 4)])
      )
    end
  end

  describe "real" do
    defn real(x), do: Nx.real(x)

    test "correctly returns real part of complex" do
      assert_equal(real(Nx.tensor(Complex.new(1, 2))), Nx.tensor(1.0))
      assert_equal(real(Nx.tensor(1)), Nx.tensor(1.0))

      assert_equal(
        real(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([1.0, 2.0])
      )
    end
  end

  describe "imag" do
    defn imag(x), do: Nx.imag(x)

    test "correctly returns imaginary part of complex" do
      assert_equal(imag(Nx.tensor(Complex.new(1, 2))), Nx.tensor(2.0))
      assert_equal(imag(Nx.tensor(1)), Nx.tensor(0.0))

      assert_equal(
        imag(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([2.0, -4.0])
      )
    end
  end

  describe "+/2" do
    defn add_two(a, b), do: a + b

    test "same shape and type" do
      assert_equal(add_two(1.0, 2.0), Nx.tensor(3.0))
      assert_equal(add_two(1, 2), Nx.tensor(3))

      assert_equal(add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4])), Nx.tensor([4, 6]))
      assert_equal(add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])), Nx.tensor([4.0, 6.0]))
    end

    test "different types" do
      tensors = [
        {1, 2},
        {1.0, 2},
        {1.0, 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), 3},
        {Nx.tensor([1, 2], type: {:u, 8}), -3},
        {Nx.tensor([1, 2], type: {:u, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:s, 8}), 3},
        {Nx.tensor([1, 2], type: {:s, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:f, 32}), 3},
        {Nx.tensor([1, 2], type: {:f, 32}), 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(-3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3.0, type: {:f, 64})}
      ]

      for {left, right} <- tensors do
        assert_all_close(add_two(left, right), evaluate(&add_two/2, [left, right]))
        assert_all_close(add_two(right, left), evaluate(&add_two/2, [right, left]))
      end
    end

    defn add_two_int(t), do: t + 2
    defn add_two_float(t), do: t + 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 16}),
        Nx.tensor([1, 2], type: {:u, 32}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 32}),
        Nx.tensor([1, 2], type: {:f, 32}),
        Nx.tensor([1, 2], type: {:f, 64})
      ]

      for t <- tensors do
        assert_equal(add_two_int(t), Nx.add(t, 2))
        assert_equal(add_two_float(t), Nx.add(t, 2.0))
      end
    end

    test "broadcast" do
      tensors = [
        {Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([1, 2]), Nx.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]])},
        {Nx.tensor([[[10], [20]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[100], [200], [300]]]),
         Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([1, 2])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[1, 2]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[[1, 2]]], [[[3, 4]]]])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[[1, 2]], [[3, 4]]])}
      ]

      for {left, right} <- tensors do
        assert_all_close(add_two(left, right), evaluate(&add_two/2, [left, right]))
        assert_all_close(add_two(right, left), evaluate(&add_two/2, [right, left]))
      end
    end

    test "names" do
      left = Nx.tensor([[10, 20]], names: [nil, :tens])
      right = Nx.tensor([[1], [2]], names: [:ones, nil])
      assert add_two(left, right).names == [:ones, :tens]
    end
  end

  describe "//2" do
    defn divide_two(a, b), do: a / b

    test "parameters" do
      tensors = [
        {1, 2},
        {1, Nx.tensor([1.0, 2.0, 3.0])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
      ]

      for {left, right} <- tensors do
        assert_all_close(divide_two(left, right), Nx.divide(left, right))
        assert_all_close(divide_two(right, left), Nx.divide(right, left))
      end
    end

    defn divide_two_int(t), do: t / 2
    defn divide_two_float(t), do: t / 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 16}),
        Nx.tensor([1, 2], type: {:u, 32}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 32}),
        Nx.tensor([1, 2], type: {:f, 32}),
        Nx.tensor([1, 2], type: {:f, 64})
      ]

      for t <- tensors do
        assert_all_close(divide_two_int(t), Nx.divide(t, 2))
        assert_all_close(divide_two_float(t), Nx.divide(t, 2.0))
      end
    end
  end

  describe "remainder" do
    defn remainder(a, b), do: Nx.remainder(a, b)

    test "integers" do
      left = Nx.tensor([-1023, 1023])
      right = Nx.tensor([[-4], [4]])
      assert Nx.shape(remainder(left, right)) == {2, 2}
      assert_all_close(remainder(left, right), Nx.remainder(left, right))
    end

    test "floats" do
      left = Nx.tensor([-8.3, -8.4, -8.5, 8.3, 8.4, 8.5])
      right = Nx.tensor([[-4.2], [-4.1], [-4.0], [4.0], [4.1], [4.2]])
      assert Nx.shape(remainder(left, right)) == {6, 6}
      assert_all_close(remainder(left, right), Nx.remainder(left, right))
    end
  end

  describe "element-wise arith operators" do
    @tensors [
      {1, 2},
      {1, Nx.tensor([1.0, 2.0, 3.0])},
      {Nx.tensor([1, 2, 3]), 1},
      {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
      {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
      {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
    ]

    defn subtract_two(a, b), do: a - b

    test "-" do
      for {left, right} <- @tensors do
        assert_all_close(subtract_two(left, right), Nx.subtract(left, right))
        assert_all_close(subtract_two(right, left), Nx.subtract(right, left))
      end
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      for {left, right} <- @tensors do
        assert_all_close(multiply_two(left, right), Nx.multiply(left, right))
        assert_all_close(multiply_two(right, left), Nx.multiply(right, left))
      end
    end

    defn unary_minus(a), do: -a

    test "negate" do
      for t <- [
            Nx.tensor([-1, 0, 1], type: {:u, 8}),
            Nx.tensor([-1, 0, 1]),
            Nx.tensor([-1.0, 1.0])
          ] do
        assert_equal(unary_minus(t), Nx.negate(t))
      end
    end

    defn max_two(a, b), do: max(a, b)

    test "max" do
      for {left, right} <- @tensors do
        assert_all_close(max_two(left, right), Nx.max(left, right))
        assert_all_close(max_two(right, left), Nx.max(right, left))
      end
    end

    defn min_two(a, b), do: min(a, b)

    test "min" do
      for {left, right} <- @tensors do
        assert_all_close(min_two(left, right), Nx.min(left, right))
        assert_all_close(min_two(right, left), Nx.min(right, left))
      end
    end

    defn power_two(a, b), do: Nx.pow(a, b)

    test "pow" do
      for {left, right} <- @tensors do
        assert_all_close(power_two(left, right), Nx.pow(left, right))
        assert_all_close(power_two(right, left), Nx.pow(right, left))
      end
    end

    defn atan2_two(a, b), do: Nx.atan2(a, b)

    test "atan2" do
      <<neg_zero::float>> = <<0x8000000000000000::64>>
      left = Nx.tensor([-1.0, neg_zero, 0.0, 1.0])
      right = Nx.tensor([[-1.0], [neg_zero], [0.0], [1.0]])

      assert_all_close(atan2_two(left, right), Nx.atan2(left, right))
      assert_all_close(atan2_two(right, left), Nx.atan2(right, left))
    end

    defn quotient_two(a, b), do: Nx.quotient(a, b)

    test "quotient" do
      int_tensors = [
        {1, 2},
        {1, Nx.tensor([1, 2, 3])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 32})}
      ]

      for {left, right} <- int_tensors do
        assert_all_close(quotient_two(left, right), Nx.quotient(left, right))
        assert_all_close(quotient_two(right, left), Nx.quotient(right, left))
      end
    end
  end

  describe "element-wise bitwise operators" do
    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[-2], [-1], [0], [1], [2]])

    defn bitwise_and(a, b), do: a &&& b

    test "bitwise_and" do
      assert Nx.shape(bitwise_and(@left, @right)) == {5, 5}
      assert_equal(bitwise_and(@left, @right), Nx.bitwise_and(@left, @right))
    end

    defn bitwise_or(a, b), do: a ||| b

    test "bitwise_or" do
      assert Nx.shape(bitwise_or(@left, @right)) == {5, 5}
      assert_equal(bitwise_or(@left, @right), Nx.bitwise_or(@left, @right))
    end

    defn bitwise_not(a), do: ~~~a

    test "bitwise_not" do
      assert Nx.shape(bitwise_not(@left)) == {5}
      assert_equal(bitwise_not(@left), Nx.bitwise_not(@left))
    end

    defn bitwise_pc(a), do: Nx.population_count(a)

    test "population_count" do
      assert Nx.shape(bitwise_pc(@left)) == {5}
      assert_equal(bitwise_pc(@left), Nx.population_count(@left))
    end

    defn bitwise_clz(a), do: Nx.count_leading_zeros(a)

    test "count_leading_zeros" do
      assert Nx.shape(bitwise_clz(@left)) == {5}
      assert_equal(bitwise_clz(@left), Nx.count_leading_zeros(@left))
    end

    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[0], [1], [2], [3], [4]])

    defn left_shift(a, b), do: a <<< b

    test "left_shift" do
      assert Nx.shape(left_shift(@left, @right)) == {5, 5}
      assert_equal(left_shift(@left, @right), Nx.left_shift(@left, @right))
    end

    @left_signed Nx.tensor([-128, -127, -2, -1, 0, 1, 2, 126, 127], type: {:s, 8})
    @right_signed Nx.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8]], type: {:s, 8})

    @left_unsigned Nx.tensor([0, 1, 2, 253, 254, 255], type: {:u, 8})
    @right_unsigned Nx.tensor([[0], [1], [2], [3], [4], [5]], type: {:u, 8})

    defn right_shift(a, b), do: a >>> b

    test "right_shift" do
      assert Nx.shape(right_shift(@left_signed, @right_signed)) == {9, 9}

      assert_equal(
        right_shift(@left_signed, @right_signed),
        Nx.right_shift(@left_signed, @right_signed)
      )

      assert Nx.shape(right_shift(@left_unsigned, @right_unsigned)) == {6, 6}

      assert_equal(
        right_shift(@left_unsigned, @right_unsigned),
        Nx.right_shift(@left_unsigned, @right_unsigned)
      )
    end
  end

  describe "exp" do
    defn exp(t), do: Nx.exp(t)

    test "computes the exp across types" do
      assert_all_close(
        Nx.tensor([1, 2, 3]) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])
      )

      assert_all_close(
        Nx.tensor([1, 2, 3], type: {:s, 8}) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
      )

      assert_all_close(
        Nx.tensor([1, 2, 3], type: {:u, 8}) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
      )

      assert_all_close(
        Nx.tensor([1.0, 2.0, 3.0]) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])
      )

      assert_all_close(
        Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> exp(),
        Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
      )
    end
  end

  describe "equal" do
    defn equal(a, b), do: Nx.equal(a, b)

    test "computes equality of scalars" do
      assert_equal(equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(0, type: {:u, 8}))
    end

    test "computes equality with broadcasting" do
      assert_equal(equal(Nx.tensor(1), Nx.tensor([1, 2, 3])), Nx.tensor([1, 0, 0], type: {:u, 8}))
    end

    test "computes equality with mixed types" do
      assert_equal(
        equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end

    defn successive_compare(y_true, y_pred) do
      y_pred
      |> Nx.equal(y_pred)
      |> Nx.equal(y_true)
    end

    test "computes successive comparisons" do
      assert_equal(successive_compare(Nx.tensor(1), Nx.tensor(1)), Nx.tensor(1, type: {:u, 8}))
    end
  end

  describe "not equal" do
    defn not_equal(a, b), do: Nx.not_equal(a, b)

    test "computes equality of scalars" do
      assert_equal(not_equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
    end

    test "computes equality with broadcasting" do
      assert_equal(
        not_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([0, 1, 1], type: {:u, 8})
      )
    end

    test "computes equality with mixed types" do
      assert_equal(
        not_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "less" do
    defn less(a, b), do: Nx.less(a, b)

    test "compares scalars" do
      assert_equal(less(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(less(Nx.tensor(1), Nx.tensor([1, 2, 3])), Nx.tensor([0, 1, 1], type: {:u, 8}))
    end

    test "compares with mixed types" do
      assert_equal(
        less(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "greater" do
    defn greater(a, b), do: Nx.greater(a, b)

    test "compares scalars" do
      assert_equal(greater(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(0, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(
        greater(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end

    test "compares with mixed types" do
      assert_equal(
        greater(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([0, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "less equal" do
    defn less_equal(a, b), do: Nx.less_equal(a, b)

    test "compares scalars" do
      assert_equal(less_equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(
        less_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end

    test "compares with mixed types" do
      assert_equal(
        less_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end
  end

  describe "greater equal" do
    defn greater_equal(a, b), do: Nx.greater_equal(a, b)

    test "compares scalars" do
      assert_equal(greater_equal(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(0, type: {:u, 8}))
    end

    test "compares with broadcasting" do
      assert_equal(
        greater_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])),
        Nx.tensor([1, 0, 0], type: {:u, 8})
      )
    end

    test "compares with mixed types" do
      assert_equal(
        greater_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])),
        Nx.tensor([1, 1, 1], type: {:u, 8})
      )
    end
  end

  describe "logical" do
    defn logical_and(a, b), do: Nx.logical_and(a, b)

    test "and" do
      assert_equal(
        logical_and(Nx.tensor([-1, 0, 1]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
          ],
          type: {:u, 8}
        )
      )

      assert_equal(
        logical_and(Nx.tensor([-1.0, 0.0, 1.0]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
          ],
          type: {:u, 8}
        )
      )
    end

    defn logical_or(a, b), do: Nx.logical_or(a, b)

    test "or" do
      assert_equal(
        logical_or(Nx.tensor([-1, 0, 1]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
          ],
          type: {:u, 8}
        )
      )

      assert_equal(
        logical_or(Nx.tensor([-1.0, 0.0, 1.0]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
          ],
          type: {:u, 8}
        )
      )
    end

    defn logical_xor(a, b), do: Nx.logical_xor(a, b)

    test "xor" do
      assert_equal(
        logical_xor(Nx.tensor([-1, 0, 1]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
          ],
          type: {:u, 8}
        )
      )

      assert_equal(
        logical_xor(Nx.tensor([-1.0, 0.0, 1.0]), Nx.tensor([[-1], [0], [1]])),
        Nx.tensor(
          [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
          ],
          type: {:u, 8}
        )
      )
    end

    defn logical_not(a), do: Nx.logical_not(a)

    test "not" do
      assert_equal(
        logical_not(Nx.tensor([-2, -1, 0, 1, 2])),
        Nx.tensor([0, 0, 1, 0, 0], type: {:u, 8})
      )
    end
  end

  describe "select" do
    defn select(pred, x, y), do: Nx.select(pred, x, y)

    test "selects one or the other with a scalar" do
      assert_equal(
        select(Nx.tensor(1), Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])),
        Nx.tensor([1, 2, 3])
      )
    end

    test "selects with type" do
      assert_equal(
        select(
          Nx.tensor(1),
          Nx.tensor([1, 2, 3], type: {:u, 8}),
          Nx.tensor([4, 5, 6], type: {:u, 8})
        ),
        Nx.tensor([1, 2, 3], type: {:u, 8})
      )

      assert_equal(
        select(
          Nx.tensor(1),
          Nx.tensor([1, 2, 3], type: {:u, 8}),
          Nx.tensor([4, 5, 6], type: {:f, 32})
        ),
        Nx.tensor([1, 2, 3], type: {:f, 32})
      )
    end

    test "selects with broadcasting" do
      assert_equal(
        select(Nx.tensor([1, 0, 1, 0, 1]), Nx.tensor([10]), Nx.tensor([1, 2, 3, 4, 5])),
        Nx.tensor([10, 2, 10, 4, 10])
      )

      assert_equal(
        select(Nx.tensor([-2, -1, 0, 1, 2]), Nx.tensor([10]), Nx.tensor([1, 2, 3, 4, 5])),
        Nx.tensor([10, 10, 3, 10, 10])
      )
    end
  end

  describe "unary float ops" do
    @int_tensor Nx.tensor([1, 2, 3])
    @float_tensor Nx.tensor([1.0, 2.0, 3.0])

    for fun <-
          [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt, :is_nan] ++
            [:is_infinity, :tan, :acosh, :asinh, :cosh, :sinh, :erf, :erfc] do
      defn_fun = :"unary_#{fun}"
      defn_var = Macro.var(defn_fun, __MODULE__)
      defn unquote(defn_fun)(t), do: Nx.unquote(fun)(t)

      test "#{fun}" do
        assert_all_close(
          unquote(defn_fun)(@float_tensor),
          evaluate(&(unquote(defn_var) / 1), [@float_tensor])
        )

        assert_all_close(
          unquote(defn_fun)(@int_tensor),
          evaluate(&(unquote(defn_var) / 1), [@int_tensor])
        )
      end
    end
  end

  describe "complex ops" do
    defn fft(t, opts \\ []), do: Nx.fft(t, opts)
    defn ifft(t, opts \\ []), do: Nx.ifft(t, opts)

    test "fft" do
      assert_all_close(
        fft(Nx.tensor([1, 1, 0, 0]), length: 5),
        ~V[2.0+0.0i 1.3090-0.9511i 0.1909-0.5877i 0.1909+0.5877i 1.3090+0.9510i]
      )

      assert_all_close(
        fft(Nx.tensor([1, 1, 0, 0, 2, 3]), length: 4),
        ~V[2.0+0.0i 1.0-1.0i 0.0+0.0i 1.0+1.0i]
      )

      assert_all_close(
        fft(Nx.tensor([1, 1, 0]), length: :power_of_two),
        ~V[2.0+0.0i 1.0-1.0i 0.0+0.0i 1.0+1.0i]
      )
    end

    test "fft - n dim tensor" do
      assert_all_close(
        fft(
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
        fft(
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
        fft(
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
        ifft(~V[5 5 5 5 5],
          length: 5
        ),
        Nx.tensor([5, 0, 0, 0, 0])
      )

      assert_all_close(
        ifft(~V[2.0+0.0i 1.0-1.0i 0.0+0.0i 1.0+1.0i 5 6], length: 4),
        Nx.tensor([1, 1, 0, 0])
      )

      assert_all_close(
        ifft(~V[2 0 0], length: :power_of_two),
        Nx.tensor([0.5, 0.5, 0.5, 0.5])
      )
    end

    test "ifft - n dim tensor" do
      assert_all_close(
        ifft(
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
        ifft(
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
        ifft(
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

  describe "unary float ops, restricted domain" do
    @int_tensor Nx.tensor([0.1, 0.5, 0.9])
    @float_tensor Nx.tensor([0.1, 0.5, 0.9])

    for fun <- [:atanh, :acos, :asin, :atan, :erf_inv] do
      defn_fun = :"unary_#{fun}"
      defn_var = Macro.var(defn_fun, __MODULE__)
      defn unquote(defn_fun)(t), do: Nx.unquote(fun)(t)

      test "#{fun}" do
        assert_all_close(
          unquote(defn_fun)(@float_tensor),
          evaluate(&(unquote(defn_var) / 1), [@float_tensor])
        )

        assert_all_close(
          unquote(defn_fun)(@int_tensor),
          evaluate(&(unquote(defn_var) / 1), [@int_tensor])
        )
      end
    end
  end

  describe "unary round+sign ops" do
    @uint_tensor Nx.tensor([0, 1, 2], type: {:u, 8})
    @sint_tensor Nx.tensor([-2, -1, 0, 1, 2])
    @float_tensor Nx.tensor([-1.5, 0.5, -0.0, 0.0, 0.5, 1.5])

    funs = [:abs, :sign, :floor, :ceil, :round]

    for fun <- funs do
      defn_fun = :"unary_#{fun}"
      defn_var = Macro.var(defn_fun, __MODULE__)
      defn unquote(defn_fun)(t), do: Nx.unquote(fun)(t)

      test "#{fun}" do
        assert_all_close(
          unquote(defn_fun)(@uint_tensor),
          evaluate(&(unquote(defn_var) / 1), [@uint_tensor])
        )

        assert_all_close(
          unquote(defn_fun)(@sint_tensor),
          evaluate(&(unquote(defn_var) / 1), [@sint_tensor])
        )

        assert_all_close(
          unquote(defn_fun)(@float_tensor),
          evaluate(&(unquote(defn_var) / 1), [@float_tensor])
        )
      end
    end
  end

  describe "as_type" do
    defn to_float(t), do: Nx.as_type(t, {:f, 32})

    test "converts tensor type" do
      assert_equal(to_float(Nx.tensor([1, 2, 3])), Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
    end

    defn generic_as_type(t, template), do: Nx.as_type(t, template.type)

    test "converts non-finite types" do
      non_finite =
        Nx.stack([Nx.Constants.infinity(), Nx.Constants.nan(), Nx.Constants.neg_infinity()])

      assert_equal(
        generic_as_type(non_finite, Nx.template({}, {:u, 8})),
        Nx.tensor([255, 0, 0], type: {:u, 8})
      )

      assert_equal(
        generic_as_type(non_finite, Nx.template({}, {:s, 16})),
        Nx.tensor([32767, 0, -32768], type: {:s, 16})
      )
    end
  end

  describe "bitcast" do
    defn bitcast_to_float(t), do: Nx.bitcast(t, {:f, 32})

    test "converts tensor type" do
      assert_equal(
        bitcast_to_float(Nx.tensor([0, 0, 0], type: {:s, 32})),
        Nx.tensor([0.0, 0.0, 0.0])
      )
    end
  end

  describe "if" do
    defn if3(a, b, c), do: if(a, do: b, else: c)

    test "one param per branch" do
      assert_equal(
        if3(Nx.tensor(0), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})),
        Nx.tensor(2, type: {:f, 32})
      )

      assert_equal(
        if3(Nx.tensor(1), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})),
        Nx.tensor(1, type: {:f, 32})
      )

      assert_equal(
        if3(Nx.tensor(2), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})),
        Nx.tensor(1, type: {:f, 32})
      )

      assert_equal(
        if3(Nx.tensor(0), Nx.tensor([1, 2]), Nx.tensor([[3], [4]])),
        Nx.tensor([[3, 3], [4, 4]])
      )

      assert_equal(
        if3(Nx.tensor(1), Nx.tensor([1, 2]), Nx.tensor([[3], [4]])),
        Nx.tensor([[1, 2], [1, 2]])
      )
    end

    defn if_params(a, b, c), do: if(a, do: b + c, else: b - c)

    test "two params per branch" do
      assert_equal(if_params(Nx.tensor(0), Nx.tensor(1), Nx.tensor(2)), Nx.tensor(-1))
      assert_equal(if_params(Nx.tensor(1), Nx.tensor(1), Nx.tensor(2)), Nx.tensor(3))
    end

    defn if_shared(a, b, c) do
      d = b + c
      if a, do: 2 * d * a, else: -1
    end

    test "shared params between pred+branch and no params" do
      assert_equal(if_shared(Nx.tensor(0), Nx.tensor(1), Nx.tensor(2)), Nx.tensor(-1))
      assert_equal(if_shared(Nx.tensor(2), Nx.tensor(1), Nx.tensor(2)), Nx.tensor(12))
    end

    defn if_tuple(a, b, c), do: if(a, do: {{a, b}, c}, else: {{c, b}, a})

    test "with tuples" do
      assert_equal(
        if_tuple(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)),
        {{Nx.tensor(20), Nx.tensor(10)}, Nx.tensor(0)}
      )

      assert_equal(
        if_tuple(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)),
        {{Nx.tensor(1), Nx.tensor(10)}, Nx.tensor(20)}
      )

      assert_equal(
        if_tuple(Nx.tensor(0), Nx.tensor(10), Nx.tensor([20, 30])),
        {{Nx.tensor([20, 30]), Nx.tensor(10)}, Nx.tensor([0, 0])}
      )

      assert_equal(
        if_tuple(Nx.tensor(1), Nx.tensor(10), Nx.tensor([20, 30])),
        {{Nx.tensor([1, 1]), Nx.tensor(10)}, Nx.tensor([20, 30])}
      )
    end

    defn if_tuple_match(a, b, c) do
      {{x, y}, z} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
      x * y - z
    end

    test "with matched tuples" do
      assert_equal(if_tuple_match(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)), Nx.tensor(200))
      assert_equal(if_tuple_match(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)), Nx.tensor(-10))
    end

    defn if_tuple_match_twice(a, b) do
      {a, b} = if(a, do: {a, b}, else: {a, b})
      {a, b} = if(a, do: {a, b}, else: {a, b})
      a + b
    end

    test "with matched tuples twice" do
      assert_equal(if_tuple_match_twice(Nx.tensor(1), Nx.tensor(1)), Nx.tensor(2))
    end

    defn if_tuple_return(a, b, c) do
      {xy, _} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
      xy
    end

    test "with return tuple" do
      assert_equal(
        if_tuple_return(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)),
        {Nx.tensor(20), Nx.tensor(10)}
      )

      assert_equal(
        if_tuple_return(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)),
        {Nx.tensor(1), Nx.tensor(10)}
      )
    end

    defn if_map(a, b, c), do: if(a, do: {%{a: a, b: b, c: 1}, c}, else: {%{a: c, b: b, c: 2}, a})

    test "with map" do
      assert_equal(
        if_map(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)),
        {%{a: Nx.tensor(20), b: Nx.tensor(10), c: Nx.tensor(2)}, Nx.tensor(0)}
      )

      assert_equal(
        if_map(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)),
        {%{a: Nx.tensor(1), b: Nx.tensor(10), c: Nx.tensor(1)}, Nx.tensor(20)}
      )

      assert_equal(
        if_map(Nx.tensor(0), Nx.tensor(10), Nx.tensor([20, 30])),
        {%{a: Nx.tensor([20, 30]), b: Nx.tensor(10), c: Nx.tensor(2)}, Nx.tensor([0, 0])}
      )

      assert_equal(
        if_map(Nx.tensor(1), Nx.tensor(10), Nx.tensor([20, 30])),
        {%{a: Nx.tensor([1, 1]), b: Nx.tensor(10), c: Nx.tensor(1)}, Nx.tensor([20, 30])}
      )
    end

    defn if_map_match(a, b, c) do
      {%{a: x, b: y}, z} = if(a, do: {%{a: a, b: b}, c}, else: {%{a: c, b: b}, a})
      x * y - z
    end

    test "with matched map" do
      assert_equal(if_map_match(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)), Nx.tensor(200))
      assert_equal(if_map_match(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)), Nx.tensor(-10))
    end

    defn grad_if_tuple(t) do
      grad(t, fn t ->
        {{a, b}, c} =
          if t > 0 do
            {{Nx.pow(t, 2), Nx.pow(t, 3)}, Nx.pow(t, 4)}
          else
            {{Nx.pow(t, 4), Nx.pow(t, 3)}, Nx.pow(t, 2)}
          end

        d = if t > 0, do: 123, else: 456

        a * b + c - d
      end)
    end

    test "grad with tuple" do
      assert_equal(grad_if_tuple(Nx.tensor(1)), Nx.tensor(9.0))
      assert_equal(grad_if_tuple(Nx.tensor(2)), Nx.tensor(112.0))
      assert_equal(grad_if_tuple(Nx.tensor(-1)), Nx.tensor(5.0))
      assert_equal(grad_if_tuple(Nx.tensor(-2)), Nx.tensor(444.0))
    end
  end

  describe "metadata" do
    defn add_with_stop_grad(a, b), do: stop_grad(Nx.add(a, b))

    test "ignores metadata nodes" do
      assert_equal(add_with_stop_grad(1, 2), Nx.tensor(3))
    end
  end

  describe "cond" do
    defn cond3(a, b, c) do
      d = Nx.sum(a)

      cond do
        Nx.all(Nx.greater(a, 0)) -> b * c * d
        Nx.all(Nx.less(a, 0)) -> b + c + d
        true -> -b - c - d
      end
    end

    test "computes cond" do
      assert_equal(cond3(Nx.tensor([-1, 0, 1]), Nx.tensor(2), Nx.tensor(3.0)), Nx.tensor(-5.0))
      assert_equal(cond3(Nx.tensor([1, 2, 3]), Nx.tensor(2), Nx.tensor(3.0)), Nx.tensor(36.0))
      assert_equal(cond3(Nx.tensor([-1, -2, -3]), Nx.tensor(2), Nx.tensor(3.0)), Nx.tensor(-1.0))
    end

    defn cond_unused_and_slice(_result, state) do
      cond do
        Nx.equal(Nx.sum(state[0..1]), 0) -> state[0]
        Nx.equal(Nx.sum(state[3..4]), 0) -> state[4]
        true -> state[2]
      end
    end

    test "computes cond with slice and unused vars" do
      assert_equal(cond_unused_and_slice(Nx.tensor(1), Nx.iota({5})), Nx.tensor(2))

      assert_equal(
        cond_unused_and_slice(Nx.tensor(1), Nx.tensor([-1, 1, 0, 1, 2])),
        Nx.tensor(-1)
      )

      assert_equal(cond_unused_and_slice(Nx.tensor(1), Nx.tensor([2, 1, 0, -1, 1])), Nx.tensor(1))
    end

    defn nested_cond(i) do
      new_i =
        if i > 0 do
          i + 1
        else
          i - 1
        end

      if new_i > 0 do
        1
      else
        0
      end
    end

    test "computes cond with cond as parameter" do
      assert_equal(nested_cond(Nx.tensor(10)), Nx.tensor(1))
      assert_equal(nested_cond(Nx.tensor(-10)), Nx.tensor(0))
    end

    defn cond_predicate(a, b) do
      cond do
        Nx.all(Nx.greater(a, 0)) -> Nx.greater(b, 0)
        true -> 0
      end
    end

    test "with predicate" do
      assert_equal(cond_predicate(Nx.tensor(1), Nx.tensor(2)), Nx.tensor(1, type: {:u, 8}))
      assert_equal(cond_predicate(Nx.tensor(1), Nx.tensor(-2)), Nx.tensor(0, type: {:u, 8}))
    end
  end

  describe "while/3" do
    defn upto10(x) do
      while x, Nx.less(x, 10) do
        x + 1
      end
    end

    test "simple" do
      assert_equal(upto10(0), Nx.tensor(10))
      assert_equal(upto10(5), Nx.tensor(10))
    end

    defn factorial_tuple(x) do
      {factorial, _} =
        while {factorial = 1.0, x}, Nx.greater(x, 1) do
          {factorial * x, x - 1}
        end

      factorial
    end

    test "factorial" do
      assert_equal(factorial_tuple(5), Nx.tensor(120.0))
      assert_equal(factorial_tuple(10.0), Nx.tensor(3_628_800.0))
    end

    defn factorial_map(x) do
      factorial = Nx.tensor(1, type: Nx.type(x))

      %{factorial: factorial} =
        while map = %{factorial: factorial, x: x}, Nx.greater(map.x, 1) do
          %{map | factorial: map.factorial * map.x, x: map.x - 1}
        end

      factorial
    end

    test "factorial map" do
      assert_equal(factorial_map(5), Nx.tensor(120))
      assert_equal(factorial_map(10.0), Nx.tensor(3_628_800.0))
    end

    defn factorial_map_input(map) do
      %{factorial: factorial} =
        while map, Nx.greater(map.x, 1) do
          %{map | factorial: map.factorial * map.x, x: map.x - 1}
        end

      factorial
    end

    test "factorial map input" do
      assert_equal(factorial_map_input(%{factorial: 1, x: 5}), Nx.tensor(120))
      assert_equal(factorial_map_input(%{factorial: 1.0, x: 10.0}), Nx.tensor(3_628_800.0))
    end

    defn while_predicate(x) do
      while {exit_condition = Nx.tensor(true), x}, exit_condition do
        res = Nx.multiply(x, x)
        {not Nx.all(res >= 0), res}
      end
    end

    test "with predicate" do
      {truth, double} = while_predicate(Nx.tensor([1.0, -2.0, 3.0]))
      assert_equal(truth, Nx.tensor(0, type: {:u, 8}))
      assert_equal(double, Nx.tensor([1.0, 4.0, 9.0]))
    end
  end

  describe "map" do
    defn map_plus(t), do: Nx.map(t, fn x -> x + 1 end)
    defn map_equal(t), do: Nx.map(t, [type: {:f, 64}], fn x -> Nx.equal(x, 1) end)
    defn map_exp(t), do: Nx.map(t, [type: {:f, 64}], fn x -> Nx.exp(x) end)

    @tag :unsupported_64_bit_op
    test "maps a function over the tensor" do
      assert_equal(map_plus(Nx.tensor([[1, 2, 3], [4, 5, 6]])), Nx.tensor([[2, 3, 4], [5, 6, 7]]))
    end

    @tag :unsupported_64_bit_op
    test "maps a function with an output type" do
      assert_equal(
        map_equal(Nx.tensor([[1, 2, 3], [4, 5, 6]])),
        Nx.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], type: {:f, 64})
      )

      assert_equal(
        map_exp(Nx.tensor([[1, 2, 3], [4, 5, 6]])),
        Nx.tensor(
          [
            [2.718281828459045, 7.38905609893065, 20.085536923187668],
            [54.598150033144236, 148.4131591025766, 403.4287934927351]
          ],
          type: {:f, 64}
        )
      )
    end

    defn map_conditional(t), do: Nx.map(t, fn x -> if x > 0, do: x, else: -x end)

    @tag :conditional_inside_map_reduce
    @tag :unsupported_64_bit_op
    test "maps a function with conditional" do
      assert_equal(
        map_conditional(Nx.tensor([-2, -1, 0, 1, 2])),
        Nx.tensor([2, 1, 0, 1, 2])
      )
    end
  end

  describe "reduce" do
    defn reduce(t), do: Nx.reduce(t, 1, fn a, b -> a * b end)
    defn reduce_keep(t), do: Nx.reduce(t, 1, [keep_axes: true], fn a, b -> a * b end)

    defn reduce_keep_2(t),
      do: Nx.reduce(t, 1, [keep_axes: true, axes: [0, 2]], fn a, b -> a * b end)

    test "computes the reduce" do
      assert_equal(Nx.tensor([1, 2, 3]) |> reduce(), Nx.tensor(6))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> reduce(), Nx.tensor(6.0))

      assert_equal(Nx.tensor([1, 2, 3], type: {:u, 8}) |> reduce(), Nx.tensor(6, type: {:u, 8}))
      assert_equal(Nx.tensor([1, 2, 3], type: {:s, 8}) |> reduce(), Nx.tensor(6, type: {:s, 8}))
      assert_equal(Nx.tensor([1, 2, 3], type: {:f, 32}) |> reduce(), Nx.tensor(6, type: {:f, 32}))
    end

    test "computes the reduce, keeping dimensions" do
      assert_equal(Nx.tensor([1, 2, 3]) |> reduce_keep(), Nx.tensor([6]))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> reduce_keep(), Nx.tensor([6.0]))

      assert_equal(
        Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) |> reduce_keep_2(),
        Nx.tensor([[[36], [14400]]])
      )
    end
  end

  describe "reduce window" do
    defn window_reduce_valid_no_stride(t),
      do: Nx.window_reduce(t, 0, {2, 2}, fn a, b -> a + b end)

    defn window_reduce_valid_stride(t),
      do: Nx.window_reduce(t, 0, {2, 2}, [strides: [2, 2]], fn a, b -> a + b end)

    defn window_reduce_same_no_stride(t),
      do: Nx.window_reduce(t, 0, {2, 2}, [padding: :same], fn a, b -> a + b end)

    defn window_reduce_same_stride(t),
      do: Nx.window_reduce(t, 0, {2, 2}, [padding: :same, strides: [2, 1]], fn a, b -> a + b end)

    defn window_reduce_general_no_stride(t),
      do: Nx.window_reduce(t, 0, {2, 2}, [padding: [{2, 1}, {1, 2}]], fn a, b -> a + b end)

    defn window_reduce_general_stride(t) do
      Nx.window_reduce(t, 0, {2, 2}, [padding: [{1, 2}, {2, 1}], strides: [2, 1]], fn a, b ->
        a + b
      end)
    end

    defn window_reduce_nd(t) do
      Nx.window_reduce(
        t,
        0,
        {1, 2, 1, 2, 1, 2},
        [padding: :same, strides: [2, 1, 1, 1, 1, 2]],
        fn a, b -> a + b end
      )
    end

    defn dilated_window_reduce(t) do
      Nx.window_reduce(
        t,
        0,
        {2, 1, 2},
        [padding: :same, strides: [1, 2, 1], window_dilations: [2, 1, 1]],
        fn a, b -> a + b end
      )
    end

    test "valid padding, no stride" do
      t = Nx.iota({6, 7})

      assert_equal(
        window_reduce_valid_no_stride(t),
        Nx.window_reduce(t, 0, {2, 2}, fn a, b -> a + b end)
      )
    end

    test "valid padding, stride" do
      t = Nx.iota({11, 10})

      assert_equal(
        window_reduce_valid_stride(t),
        Nx.window_reduce(t, 0, {2, 2}, [strides: [2, 2]], fn a, b -> a + b end)
      )
    end

    test "same padding, no stride" do
      t = Nx.iota({3, 3})

      assert_equal(
        window_reduce_same_no_stride(t),
        Nx.window_reduce(t, 0, {2, 2}, [padding: :same], fn a, b -> a + b end)
      )
    end

    test "same padding, stride" do
      t = Nx.iota({8, 8})

      assert_equal(
        window_reduce_same_stride(t),
        Nx.window_reduce(t, 0, {2, 2}, [padding: :same, strides: [2, 1]], fn a, b ->
          a + b
        end)
      )
    end

    test "general padding, no stride" do
      t = Nx.iota({3, 3})

      assert_equal(
        window_reduce_general_no_stride(t),
        Nx.window_reduce(t, 0, {2, 2}, [padding: [{2, 1}, {1, 2}]], fn a, b -> a + b end)
      )
    end

    test "general padding, stride" do
      t = Nx.iota({7, 7})

      assert_equal(
        window_reduce_general_stride(t),
        Nx.window_reduce(
          t,
          0,
          {2, 2},
          [padding: [{1, 2}, {2, 1}], strides: [2, 1]],
          fn a, b -> a + b end
        )
      )
    end

    test "n-d reduce window" do
      t = Nx.iota({4, 2, 4, 3, 1, 3})

      assert_equal(
        window_reduce_nd(t),
        Nx.window_reduce(
          t,
          0,
          {1, 2, 1, 2, 1, 2},
          [padding: :same, strides: [2, 1, 1, 1, 1, 2]],
          fn a, b -> a + b end
        )
      )
    end

    @tag :unsupported_dilated_window_reduce
    test "computes a dilated reduce window" do
      t = Nx.iota({6, 4, 3})

      assert_equal(
        dilated_window_reduce(t),
        Nx.window_reduce(
          t,
          0,
          {2, 1, 2},
          [padding: :same, strides: [1, 2, 1], window_dilations: [2, 1, 1]],
          fn a, b -> a + b end
        )
      )
    end
  end

  describe "window_scatter_min/max" do
    defn window_scatter_max_no_padding(t) do
      Nx.window_scatter_max(
        t,
        Nx.tensor([[2, 6], [3, 1]]),
        0,
        {2, 3},
        padding: :valid,
        strides: [2, 3]
      )
    end

    @tag :unsupported_64_bit_op
    test "window_scatter_max produces the same result as Nx with no padding" do
      x =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      lhs = window_scatter_max_no_padding(x)

      rhs =
        Nx.window_scatter_max(
          x,
          Nx.tensor([[2, 6], [3, 1]]),
          0,
          {2, 3},
          padding: :valid,
          strides: [2, 3]
        )

      assert_all_close(lhs, rhs)
    end

    defn window_scatter_min_no_padding(t) do
      Nx.window_scatter_min(
        t,
        Nx.tensor([[2, 6], [3, 1]]),
        0,
        {2, 3},
        padding: :valid,
        strides: [2, 3]
      )
    end

    @tag :unsupported_64_bit_op
    test "window_scatter_min produces the same result as Nx with no padding" do
      x =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      lhs = window_scatter_min_no_padding(x)

      rhs =
        Nx.window_scatter_min(
          x,
          Nx.tensor([[2, 6], [3, 1]]),
          0,
          {2, 3},
          padding: :valid,
          strides: [2, 3]
        )

      assert_all_close(lhs, rhs)
    end
  end

  describe "indexed_add" do
    defn indexed_add(t, i, u) do
      Nx.indexed_add(t, i, u)
    end

    test "indexed_add works for multi-dim tensor" do
      target = Nx.broadcast(0, {2, 3, 4})

      indices =
        Nx.tensor([
          [0, 0, 0],
          [0, 0, 1],
          [0, 0, 0],
          [0, 1, 1],
          [1, 2, 2],
          [1, 2, 3],
          [1, 0, 3]
        ])

      updates =
        Nx.tensor([
          1,
          1,
          -1,
          2,
          -1,
          3,
          4
        ])

      assert_equal(
        Nx.tensor([
          [
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, -1, 3]
          ]
        ]),
        indexed_add(target, indices, updates)
      )
    end

    test "indexed_add handles different input types" do
      target = Nx.tensor([0])
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1])

      assert_equal(indexed_add(target, indices, updates), Nx.tensor([1], type: {:s, 64}))

      target = Nx.tensor([0])
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1.0])

      assert_equal(indexed_add(target, indices, updates), Nx.tensor([1.0], type: {:f, 32}))

      target = Nx.tensor([0.0])
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1])

      assert_equal(indexed_add(target, indices, updates), Nx.tensor([1.0], type: {:f, 32}))

      target = Nx.tensor([0.0], type: {:f, 64})
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1.0], type: {:f, 32})

      assert_equal(indexed_add(target, indices, updates), Nx.tensor([1.0], type: {:f, 64}))
    end
  end

  describe "indexed_put" do
    defn indexed_put(t, i, u) do
      Nx.indexed_put(t, i, u)
    end

    test "indexed_add works for multi-dim tensor" do
      target = Nx.broadcast(0, {2, 3, 4})

      indices =
        Nx.tensor([
          [0, 0, 0],
          [0, 0, 1],
          [0, 0, 0],
          [0, 1, 1],
          [1, 2, 2],
          [1, 2, 3],
          [1, 0, 3]
        ])

      updates =
        Nx.tensor([
          1,
          1,
          -1,
          2,
          -1,
          3,
          4
        ])

      assert_equal(
        Nx.tensor([
          [
            [-1, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, -1, 3]
          ]
        ]),
        indexed_put(target, indices, updates)
      )
    end

    test "indexed_put handles different input types" do
      target = Nx.tensor([0])
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1])

      assert_equal(indexed_put(target, indices, updates), Nx.tensor([1], type: {:s, 64}))

      target = Nx.tensor([0])
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1.0])

      assert_equal(indexed_put(target, indices, updates), Nx.tensor([1.0], type: {:f, 32}))

      target = Nx.tensor([0.0])
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1])

      assert_equal(indexed_put(target, indices, updates), Nx.tensor([1.0], type: {:f, 32}))

      target = Nx.tensor([0.0], type: {:f, 64})
      indices = Nx.tensor([[0]])
      updates = Nx.tensor([1.0], type: {:f, 32})

      assert_equal(indexed_put(target, indices, updates), Nx.tensor([1.0], type: {:f, 64}))
    end
  end

  describe "all" do
    defn all(t), do: Nx.all(t)
    defn all_axis_0(t), do: Nx.all(t, axes: [0])
    defn all_axis_1(t), do: Nx.all(t, axes: [1])
    defn all_keep_axes(t), do: Nx.all(t, keep_axes: true)

    test "computes the bitwise and across types" do
      assert_equal(all(Nx.tensor([1, 2, 3])), Nx.tensor(1, type: {:u, 8}))
      assert_equal(all(Nx.tensor([0, 1, 2])), Nx.tensor(0, type: {:u, 8}))
      assert_equal(all(Nx.tensor([0.0, 1.0, 2.0])), Nx.tensor(0, type: {:u, 8}))
    end

    test "computes the bitwise and on given axes" do
      assert_equal(
        all_axis_0(Nx.tensor([[-1, 0, 1], [2, 3, 4]])),
        Nx.tensor([1, 0, 1], type: {:u, 8})
      )

      assert_equal(
        all_axis_1(Nx.tensor([[-1, 0, 1], [2, 3, 4]])),
        Nx.tensor([0, 1], type: {:u, 8})
      )
    end

    test "computes the bitwise and keeps axes" do
      assert_equal(
        all_keep_axes(Nx.tensor([[-1, 0, 1], [2, 3, 4]])),
        Nx.tensor([[0]], type: {:u, 8})
      )
    end
  end

  describe "any" do
    defn any(t), do: Nx.any(t)
    defn any_axis_0(t), do: Nx.any(t, axes: [0])
    defn any_axis_1(t), do: Nx.any(t, axes: [1])

    test "computes the bitwise and across types" do
      assert_equal(any(Nx.tensor([-1, 0, 1])), Nx.tensor(1, type: {:u, 8}))
      assert_equal(any(Nx.tensor([0, 0, 0])), Nx.tensor(0, type: {:u, 8}))
      assert_equal(any(Nx.tensor([-1.0, 0.0, 1.0])), Nx.tensor(1, type: {:u, 8}))
    end

    test "computes the bitwise and on given axes" do
      assert_equal(
        any_axis_0(Nx.tensor([[0, 1, 0], [0, 1, 2]])),
        Nx.tensor([0, 1, 1], type: {:u, 8})
      )

      assert_equal(
        any_axis_1(Nx.tensor([[0, 1, 0], [0, 1, 2]])),
        Nx.tensor([1, 1], type: {:u, 8})
      )
    end
  end

  describe "sum" do
    defn sum(t), do: Nx.sum(t)

    test "computes the sum across types" do
      assert_equal(Nx.tensor([1, 2, 3]) |> sum(), Nx.tensor(6))
      assert_equal(Nx.tensor([1, 2, 3], type: {:s, 8}) |> sum(), Nx.tensor(6))
      assert_equal(Nx.tensor([1, 2, 3], type: {:u, 8}) |> sum(), Nx.tensor(6, type: {:u, 64}))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> sum(), Nx.tensor(6.0))

      assert_equal(
        Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> sum(),
        Nx.tensor(6, type: {:f, 32})
      )
    end

    defn sum_pos_axis(t), do: Nx.sum(t, axes: [1])
    defn sum_neg_axis(t), do: Nx.sum(t, axes: [-3])
    defn sum_pos_neg_axis(t), do: Nx.sum(t, axes: [1, -3])

    test "computes the sum on a given axis" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      assert_equal(sum_pos_axis(t), Nx.sum(t, axes: [1]))
      assert_equal(sum_neg_axis(t), Nx.sum(t, axes: [-3]))
      assert_equal(sum_pos_neg_axis(t), Nx.sum(t, axes: [1, -3]))
    end

    defn sum_equal(t), do: Nx.sum(Nx.equal(t, 1.0))

    test "does not overflow" do
      assert_equal(sum_equal(Nx.tensor(1)), Nx.tensor(1, type: {:u, 64}))
      assert_equal(sum_equal(Nx.tensor([1, 1, 1])), Nx.tensor(3, type: {:u, 64}))
      assert_equal(sum_equal(Nx.tensor([1, 2, 3])), Nx.tensor(1, type: {:u, 64}))
    end

    defn sum_keep(t), do: Nx.sum(t, keep_axes: true)
    defn sum_keep_2(t), do: Nx.sum(t, axes: [0, 2], keep_axes: true)

    test "keeps dimensions if keep_axes" do
      assert_equal(Nx.tensor([1, 2, 3]) |> sum_keep(), Nx.tensor([6]))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> sum_keep(), Nx.tensor([6.0]))

      assert_equal(
        Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) |> sum_keep_2(),
        Nx.tensor([[[12], [30]]])
      )
    end
  end

  describe "product" do
    defn product(t), do: Nx.product(t)

    test "computes the product across types" do
      assert_equal(Nx.tensor([1, 2, 3]) |> product(), Nx.tensor(6))
      assert_equal(Nx.tensor([1, 2, 3], type: {:s, 8}) |> product(), Nx.tensor(6))
      assert_equal(Nx.tensor([1, 2, 3], type: {:u, 8}) |> product(), Nx.tensor(6, type: {:u, 64}))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> product(), Nx.tensor(6.0))

      assert_equal(
        Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> product(),
        Nx.tensor(6, type: {:f, 32})
      )
    end

    defn product_pos_axis(t), do: Nx.product(t, axes: [1])
    defn product_neg_axis(t), do: Nx.product(t, axes: [-3])
    defn product_pos_neg_axis(t), do: Nx.product(t, axes: [1, -3])

    test "computes the sum on a given axis" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      assert_equal(product_pos_axis(t), Nx.product(t, axes: [1]))
      assert_equal(product_neg_axis(t), Nx.product(t, axes: [-3]))
      assert_equal(product_pos_neg_axis(t), Nx.product(t, axes: [1, -3]))
    end

    defn product_equal(t), do: Nx.product(Nx.equal(t, 1.0))

    test "does not overflow" do
      assert_equal(product_equal(Nx.tensor(1)), Nx.tensor(1, type: {:u, 64}))
      assert_equal(product_equal(Nx.tensor([1, 1, 1])), Nx.tensor(1, type: {:u, 64}))
      assert_equal(product_equal(Nx.tensor([1, 2, 3])), Nx.tensor(0, type: {:u, 64}))
    end

    defn product_keep(t), do: Nx.product(t, keep_axes: true)
    defn product_keep_2(t), do: Nx.product(t, axes: [0, 2], keep_axes: true)

    test "keeps dimensions if keep_axes" do
      assert_equal(Nx.tensor([1, 2, 3]) |> product_keep(), Nx.tensor([6]))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> product_keep(), Nx.tensor([6.0]))

      assert_equal(
        Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) |> product_keep_2(),
        Nx.tensor([[[36], [14400]]])
      )
    end
  end

  describe "mean" do
    defn mean(t), do: Nx.mean(t)

    test "computes mean without axis" do
      assert_equal(mean(Nx.tensor(42)), Nx.tensor(42.0))
      assert_equal(mean(Nx.tensor([1, 2, 3])), Nx.tensor(2.0))
      assert_equal(mean(Nx.tensor([1, 2, 3], type: {:u, 8})), Nx.tensor(2.0, type: {:f, 32}))
    end

    defn mean_over_single_axis(t), do: Nx.mean(t, axes: [0])

    test "computes mean over a single axis" do
      assert_equal(mean_over_single_axis(Nx.tensor([1, 2, 3])), Nx.tensor(2.0))

      assert_equal(
        mean_over_single_axis(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])),
        Nx.tensor([
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]
        ])
      )
    end

    defn mean_over_multiple_axes(t), do: Nx.mean(t, axes: [0, 2])

    test "computes mean over multiple axes" do
      assert_equal(
        mean_over_multiple_axes(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])),
        Nx.tensor([5.0, 8.0])
      )
    end

    defn mean_over_negative_axis(t), do: Nx.mean(t, axes: [-1])

    test "computes mean over negative axes" do
      assert_equal(
        mean_over_negative_axis(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])),
        Nx.tensor([[2.0, 5.0], [8.0, 11.0]])
      )
    end

    defn mean_equal(t), do: Nx.mean(Nx.equal(t, 1.0))

    test "does not overflow" do
      assert_equal(mean_equal(Nx.tensor(1)), Nx.tensor(1.0))
      assert_equal(mean_equal(Nx.tensor([1, 1, 1])), Nx.tensor(1.0))
      assert_equal(mean_equal(Nx.tensor([1, 2, 3])), Nx.tensor(0.3333333333333333))
    end

    defn mean_keep(t), do: Nx.mean(t, keep_axes: true)
    defn mean_keep_2(t), do: Nx.mean(t, axes: [0, 2], keep_axes: true)

    test "keeps dimensions if keep_axes" do
      assert_equal(Nx.tensor([1, 2, 3]) |> mean_keep(), Nx.tensor([2.0]))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> mean_keep(), Nx.tensor([2.0]))

      assert_equal(
        Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) |> mean_keep_2(),
        Nx.tensor([[[2.0], [5.0]]])
      )
    end
  end

  describe "reduce_max" do
    defn reduce_max(t), do: Nx.reduce_max(t)

    test "computes the maximum across types" do
      assert_equal(Nx.tensor([1, 2, 3]) |> reduce_max(), Nx.tensor(3))

      assert_equal(
        Nx.tensor([1, 2, 3], type: {:s, 8}) |> reduce_max(),
        Nx.tensor(3, type: {:s, 8})
      )

      assert_equal(
        Nx.tensor([1, 2, 3], type: {:u, 8}) |> reduce_max(),
        Nx.tensor(3, type: {:u, 8})
      )

      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> reduce_max(), Nx.tensor(3.0))

      assert_equal(
        Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> reduce_max(),
        Nx.tensor(3, type: {:f, 32})
      )
    end

    defn reduce_max_pos_axis(t), do: Nx.reduce_max(t, axes: [1])
    defn reduce_max_neg_axis(t), do: Nx.reduce_max(t, axes: [-3])
    defn reduce_max_pos_neg_axis(t), do: Nx.reduce_max(t, axes: [1, -3])

    test "computes the max on a given axis" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      assert_equal(reduce_max_pos_axis(t), Nx.reduce_max(t, axes: [1]))
      assert_equal(reduce_max_neg_axis(t), Nx.reduce_max(t, axes: [-3]))
      assert_equal(reduce_max_pos_neg_axis(t), Nx.reduce_max(t, axes: [1, -3]))
    end

    defn reduce_max_keep(t), do: Nx.reduce_max(t, keep_axes: true)
    defn reduce_max_keep_2(t), do: Nx.reduce_max(t, axes: [0, 2], keep_axes: true)

    test "keeps dimensions if keep_axes" do
      assert_equal(Nx.tensor([1, 2, 3]) |> reduce_max_keep(), Nx.tensor([3]))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> reduce_max_keep(), Nx.tensor([3.0]))

      assert_equal(
        Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) |> reduce_max_keep_2(),
        Nx.tensor([[[3], [6]]])
      )
    end
  end

  describe "reduce_min" do
    defn reduce_min(t), do: Nx.reduce_min(t)

    test "computes the minimum across types" do
      assert_equal(Nx.tensor([1, 2, 3]) |> reduce_min(), Nx.tensor(1))

      assert_equal(
        Nx.tensor([1, 2, 3], type: {:s, 8}) |> reduce_min(),
        Nx.tensor(1, type: {:s, 8})
      )

      assert_equal(
        Nx.tensor([1, 2, 3], type: {:u, 8}) |> reduce_min(),
        Nx.tensor(1, type: {:u, 8})
      )

      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> reduce_min(), Nx.tensor(1.0))

      assert_equal(
        Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> reduce_min(),
        Nx.tensor(1, type: {:f, 32})
      )
    end

    test "computes the minimum across nan" do
      assert_equal(Nx.tensor([:nan, :nan, :nan]) |> reduce_min(), Nx.tensor(:nan))
    end

    defn reduce_min_pos_axis(t), do: Nx.reduce_min(t, axes: [1])
    defn reduce_min_neg_axis(t), do: Nx.reduce_min(t, axes: [-3])
    defn reduce_min_pos_neg_axis(t), do: Nx.reduce_min(t, axes: [1, -3])

    test "computes the min on a given axis" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      assert_equal(reduce_min_pos_axis(t), Nx.reduce_min(t, axes: [1]))
      assert_equal(reduce_min_neg_axis(t), Nx.reduce_min(t, axes: [-3]))
      assert_equal(reduce_min_pos_neg_axis(t), Nx.reduce_min(t, axes: [1, -3]))
    end

    defn reduce_min_keep(t), do: Nx.reduce_min(t, keep_axes: true)
    defn reduce_min_keep_2(t), do: Nx.reduce_min(t, axes: [0, 2], keep_axes: true)

    test "keeps dimensions if keep_axes" do
      assert_equal(Nx.tensor([1, 2, 3]) |> reduce_min_keep(), Nx.tensor([1]))
      assert_equal(Nx.tensor([1.0, 2.0, 3.0]) |> reduce_min_keep(), Nx.tensor([1.0]))

      assert_equal(
        Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) |> reduce_min_keep_2(),
        Nx.tensor([[[1], [4]]])
      )
    end
  end

  describe "argmax/argmin" do
    defn argmax(t), do: Nx.argmax(t)
    defn argmin(t), do: Nx.argmin(t)
    defn argmax_axis(t), do: Nx.argmax(t, axis: 1)
    defn argmin_axis(t), do: Nx.argmin(t, axis: 1)
    defn argmax_high(t), do: Nx.argmax(t, axis: 1, tie_break: :high)
    defn argmin_high(t), do: Nx.argmin(t, axis: 1, tie_break: :high)
    defn argmax_keep_axis(t), do: Nx.argmax(t, axis: 1, keep_axis: true)
    defn argmin_keep_axis(t), do: Nx.argmin(t, axis: 1, keep_axis: true)

    test "computes the argmax across types" do
      assert_equal(argmax(Nx.tensor([1, 2, 3])), Nx.tensor(2))
      assert_equal(argmax(Nx.tensor([1, 2, 3], type: {:s, 8})), Nx.tensor(2))
      assert_equal(argmax(Nx.tensor([1, 2, 3], type: {:u, 8})), Nx.tensor(2))
      assert_equal(argmax(Nx.tensor([1.0, 2.0, 3.0])), Nx.tensor(2))
      assert_equal(argmax(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})), Nx.tensor(2))
      assert_equal(argmax(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16})), Nx.tensor(2))
      assert_equal(argmax(Nx.tensor([[1, 2, 3], [4, 5, 6]])), Nx.tensor(5))
    end

    test "computes the argmin across types" do
      assert_equal(argmin(Nx.tensor([1, 2, 3])), Nx.tensor(0))
      assert_equal(argmin(Nx.tensor([1, 2, 3], type: {:s, 8})), Nx.tensor(0))
      assert_equal(argmin(Nx.tensor([1, 2, 3], type: {:u, 8})), Nx.tensor(0))
      assert_equal(argmin(Nx.tensor([1.0, 2.0, 3.0])), Nx.tensor(0))
      assert_equal(argmin(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})), Nx.tensor(0))
      assert_equal(argmin(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16})), Nx.tensor(0))
      assert_equal(argmin(Nx.tensor([[1, 2, 3], [4, 5, 6]])), Nx.tensor(0))
    end

    test "computes the argmax on an axis" do
      assert_equal(
        argmax_axis(Nx.tensor([[[1, 1, 1], [1, 1, 3]], [[6, 2, 3], [2, 8, 3]]])),
        Nx.tensor([[0, 0, 1], [0, 1, 0]])
      )
    end

    test "computes the argmin on an axis" do
      assert_equal(
        argmin_axis(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])),
        Nx.tensor([[1, 1, 0], [1, 0, 0]])
      )
    end

    test "computes argmax with tie_break: :high" do
      assert_equal(argmax_axis(Nx.tensor([[1, 2, 2], [1, 2, 2]])), Nx.tensor([1, 1]))
      assert_equal(argmax_high(Nx.tensor([[1, 2, 2], [1, 2, 2]])), Nx.tensor([2, 2]))
    end

    test "computes argmax with keep_axis: true" do
      assert_equal(
        argmax_keep_axis(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])),
        Nx.tensor([
          [[0, 0, 0]],
          [[0, 1, 0]]
        ])
      )
    end

    test "computes argmin with keep_axis: true" do
      assert_equal(
        argmin_keep_axis(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])),
        Nx.tensor([
          [[1, 1, 0]],
          [[1, 0, 0]]
        ])
      )
    end
  end

  describe "window sum" do
    defn window_sum1(t), do: Nx.window_sum(t, {1, 2, 1})

    defn window_sum2(t),
      do: Nx.window_sum(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])

    defn window_sum3(t),
      do: Nx.window_sum(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

    defn dilated_window_sum(t) do
      Nx.window_sum(t, {3, 2, 1}, strides: [1, 1, 1], padding: :same, window_dilations: [1, 2, 2])
    end

    test "computes the sum of a window" do
      assert_equal(
        window_sum1(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[5, 7, 9]], [[5, 7, 9]]])
      )

      assert_equal(
        window_sum2(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[0, 0], [0, 18]], [[0, 0], [0, 9]]])
      )

      assert_equal(
        window_sum3(
          Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
        ),
        Nx.tensor([
          [[0.0, 4.0, 2.0, 3.0, 0.0], [0.0, 2.0, 5.0, 6.5, 0.0]],
          [[0.0, 1.2, 2.2, 3.2, 0.0], [0.0, 4.0, 5.0, 6.2, 0.0]]
        ])
      )
    end

    @tag :unsupported_dilated_window_reduce
    test "computes the sum of a dilated window" do
      t = Nx.iota({8, 10, 12})

      assert_equal(
        dilated_window_sum(t),
        Nx.window_sum(t, {3, 2, 1},
          strides: [1, 1, 1],
          padding: :same,
          window_dilations: [1, 2, 2]
        )
      )
    end
  end

  describe "window mean" do
    defn window_mean1(t), do: Nx.window_mean(t, {1, 2, 1})

    defn window_mean2(t),
      do: Nx.window_mean(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])

    defn window_mean3(t),
      do: Nx.window_mean(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

    defn dilated_window_mean(t) do
      Nx.window_mean(t, {3, 2, 1},
        strides: [1, 1, 1],
        padding: :same,
        window_dilations: [1, 2, 2]
      )
    end

    test "computes the mean of a window" do
      assert_equal(
        window_mean1(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[2.5, 3.5, 4.5]], [[2.5, 3.5, 4.5]]])
      )

      assert_equal(
        window_mean2(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[0, 0], [0, 4.5]], [[0, 0], [0, 2.25]]])
      )

      assert_equal(
        window_mean3(
          Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
        ),
        Nx.tensor([
          [[0.0, 2.0, 1.0, 1.5, 0.0], [0.0, 1.0, 2.5, 3.25, 0.0]],
          [[0.0, 0.6, 1.1, 1.6, 0.0], [0.0, 2.0, 2.5, 3.1, 0.0]]
        ])
      )
    end

    @tag :unsupported_dilated_window_reduce
    test "computes the mean of a dilated window" do
      t = Nx.iota({8, 10, 12})
      lhs = dilated_window_mean(t)

      rhs =
        Nx.window_mean(t, {3, 2, 1},
          strides: [1, 1, 1],
          padding: :same,
          window_dilations: [1, 2, 2]
        )

      assert_all_close(lhs, rhs)
    end
  end

  describe "window max" do
    defn window_max1(t), do: Nx.window_max(t, {1, 2, 1})

    defn window_max2(t),
      do: Nx.window_max(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])

    defn window_max3(t),
      do: Nx.window_max(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

    defn dilated_window_max(t) do
      Nx.window_max(t, {3, 2, 1}, strides: [1, 1, 1], padding: :same, window_dilations: [1, 2, 2])
    end

    test "computes the max of a window" do
      assert_equal(
        window_max1(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[4, 5, 6]], [[4, 5, 6]]])
      )

      assert_equal(
        window_max2(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([
          [
            [-9_223_372_036_854_775_808, -9_223_372_036_854_775_808],
            [-9_223_372_036_854_775_808, 6]
          ],
          [
            [-9_223_372_036_854_775_808, -9_223_372_036_854_775_808],
            [-9_223_372_036_854_775_808, 6]
          ]
        ])
      )

      assert_equal(
        window_max3(
          Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
        ),
        Nx.tensor([
          [
            [:neg_infinity, 4.0, 2.0, 3.0, :neg_infinity],
            [:neg_infinity, 2.0, 5.0, 6.5, :neg_infinity]
          ],
          [
            [:neg_infinity, 1.2, 2.2, 3.2, :neg_infinity],
            [:neg_infinity, 4.0, 5.0, 6.2, :neg_infinity]
          ]
        ])
      )
    end

    @tag :unsupported_dilated_window_reduce
    test "computes the max of a dilated window" do
      t = Nx.iota({8, 10, 12}, type: {:f, 64})

      assert_equal(
        dilated_window_max(t),
        Nx.window_max(t, {3, 2, 1},
          strides: [1, 1, 1],
          padding: :same,
          window_dilations: [1, 2, 2]
        )
      )
    end
  end

  describe "window min" do
    defn window_min0(t), do: Nx.window_min(t, {2})
    defn window_min1(t), do: Nx.window_min(t, {1, 2, 1})

    defn window_min2(t),
      do: Nx.window_min(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])

    defn window_min3(t),
      do: Nx.window_min(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

    defn dilated_window_min(t) do
      Nx.window_min(t, {3, 2, 1}, strides: [1, 1, 1], padding: :same, window_dilations: [1, 2, 2])
    end

    test "computes the min of a window" do
      assert_equal(
        window_min1(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[1, 2, 3]], [[1, 2, 3]]])
      )

      assert_equal(
        window_min2(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([
          [
            [9_223_372_036_854_775_807, 9_223_372_036_854_775_807],
            [9_223_372_036_854_775_807, 3]
          ],
          [
            [9_223_372_036_854_775_807, 9_223_372_036_854_775_807],
            [9_223_372_036_854_775_807, 3]
          ]
        ])
      )

      assert_equal(
        window_min3(
          Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
        ),
        Nx.tensor([
          [
            [:infinity, 4.0, 2.0, 3.0, :infinity],
            [:infinity, 2.0, 5.0, 6.5, :infinity]
          ],
          [
            [:infinity, 1.2, 2.2, 3.2, :infinity],
            [:infinity, 4.0, 5.0, 6.2, :infinity]
          ]
        ])
      )
    end

    @tag :unsupported_dilated_window_reduce
    test "computes the min of a dilated window" do
      t = Nx.iota({8, 10, 12})

      assert_equal(
        dilated_window_min(t),
        Nx.window_min(t, {3, 2, 1},
          strides: [1, 1, 1],
          padding: :same,
          window_dilations: [1, 2, 2]
        )
      )
    end

    test "computes the minimum across nan" do
      assert_equal(Nx.tensor([:nan, :nan, :nan]) |> window_min0(), Nx.tensor([:nan, :nan]))
    end
  end

  describe "window product" do
    defn window_product1(t), do: Nx.window_product(t, {1, 2, 1})

    defn window_product2(t),
      do: Nx.window_product(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])

    defn window_product3(t),
      do: Nx.window_product(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])

    defn dilated_window_product(t) do
      Nx.window_product(t, {3, 2, 1},
        strides: [1, 1, 1],
        padding: :same,
        window_dilations: [1, 2, 2]
      )
    end

    test "computes the product of a window" do
      assert_equal(
        window_product1(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[4, 10, 18]], [[4, 10, 18]]])
      )

      assert_equal(
        window_product2(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])),
        Nx.tensor([[[1, 1], [1, 324]], [[1, 1], [1, 18]]])
      )

      assert_equal(
        window_product3(
          Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
        ),
        Nx.tensor([
          [[1.0, 4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 5.0, 6.5, 1.0]],
          [[1.0, 1.2, 2.2, 3.2, 1.0], [1.0, 4.0, 5.0, 6.2, 1.0]]
        ])
      )
    end

    @tag :unsupported_dilated_window_reduce
    test "computes the product of a dilated window" do
      t = Nx.iota({8, 10, 12})

      assert_equal(
        dilated_window_product(t),
        Nx.window_product(t, {3, 2, 1},
          strides: [1, 1, 1],
          padding: :same,
          window_dilations: [1, 2, 2]
        )
      )
    end
  end

  describe "dot product" do
    defn dot(a, b), do: Nx.dot(a, b)

    test "computes the dot product of scalars" do
      assert_equal(dot(Nx.tensor(2), Nx.tensor(2)), Nx.tensor(4))
      assert_equal(dot(Nx.tensor(2.0), Nx.tensor(2.0)), Nx.tensor(4.0))
      assert_equal(dot(Nx.tensor(-2.0), Nx.tensor(-2)), Nx.tensor(4.0))
    end

    test "computes the dot product of vectors" do
      assert_equal(
        dot(Nx.tensor([1, 2, 3], type: {:s, 32}), Nx.tensor([4, 5, 6], type: {:s, 32})),
        Nx.tensor(32, type: {:s, 32})
      )

      assert_equal(
        dot(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}), Nx.tensor([4, 5, 6])),
        Nx.tensor(32.0)
      )

      assert_equal(dot(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([4.0, 5.0, 6.0])), Nx.tensor(32.0))
    end

    test "computes the dot product of matrices" do
      assert_equal(
        dot(
          Nx.tensor([[1, 2, 3], [4, 5, 6]], type: {:s, 32}),
          Nx.tensor([[7, 8], [9, 10], [11, 12]], type: {:s, 32})
        ),
        Nx.tensor([[58, 64], [139, 154]], type: {:s, 32})
      )

      assert_equal(
        dot(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        ),
        Nx.tensor([[58.0, 64.0], [139.0, 154.0]])
      )

      assert_equal(
        dot(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[7, 8], [9, 10], [11, 12]])
        ),
        Nx.tensor([[58.0, 64.0], [139.0, 154.0]])
      )
    end

    test "computes the dot product of tensors" do
      assert_equal(
        dot(
          Nx.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            type: {:s, 32}
          ),
          Nx.tensor(
            [[[1, 2, 3], [3, 4, 5], [5, 6, 7]]],
            type: {:s, 32}
          )
        ),
        Nx.tensor(
          [
            [[[22, 28, 34]], [[49, 64, 79]], [[76, 100, 124]]],
            [[[22, 28, 34]], [[49, 64, 79]], [[76, 100, 124]]]
          ],
          type: {:s, 32}
        )
      )
    end

    defn batched_dot(t1, t2), do: Nx.dot(t1, [1], [0], t2, [1], [0])

    test "computes a batched dot product" do
      assert_equal(
        batched_dot(Nx.iota({3, 2, 3}, type: {:f, 32}), Nx.iota({3, 2, 2}, type: {:f, 32})),
        Nx.tensor([
          [[6.0, 9.0], [8.0, 13.0], [10.0, 17.0]],
          [[78.0, 93.0], [88.0, 105.0], [98.0, 117.0]],
          [[246.0, 273.0], [264.0, 293.0], [282.0, 313.0]]
        ])
      )
    end

    defn general_dot(t1, t2), do: Nx.dot(t1, [0, 1], [], t2, [1, 2], [])

    test "computes a general dot product" do
      assert_equal(
        general_dot(Nx.iota({4, 5, 2}, type: {:f, 32}), Nx.iota({2, 4, 5}, type: {:f, 32})),
        Nx.tensor([[4940.0, 12540.0], [5130.0, 13130.0]])
      )
    end
  end

  describe "convolution" do
    defn conv_valid_no_stride(inp, kernel), do: Nx.conv(inp, kernel)

    defn conv_valid_stride(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [2, 2], padding: :valid)

    defn conv_same_no_stride(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [1, 1], padding: :same)

    defn conv_same_stride(inp, kernel), do: Nx.conv(inp, kernel, strides: [3, 3], padding: :same)

    defn conv_general_no_stride(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [1, 1], padding: [{-1, 2}, {3, -1}])

    defn conv_general_stride(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [2, 1], padding: [{2, 2}, {-2, 4}])

    defn conv_3d(inp, kernel), do: Nx.conv(inp, kernel, strides: [1, 2, 1], padding: :same)

    defn dilated_conv(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [1, 1], padding: :same, kernel_dilation: [1, 2])

    defn dilated_input_conv(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [1, 1], padding: :same, input_dilation: [2, 1])

    defn dilated_input_kernel_conv(inp, kernel),
      do:
        Nx.conv(inp, kernel,
          strides: [2, 1],
          padding: :same,
          kernel_dilation: [2, 1],
          input_dilation: [1, 2]
        )

    defn grouped_conv_valid_no_stride(inp, kernel),
      do: Nx.conv(inp, kernel, strides: 1, padding: :valid, feature_group_size: 2)

    defn grouped_conv_same_stride(inp, kernel),
      do: Nx.conv(inp, kernel, strides: [2, 1, 2], padding: :same, feature_group_size: 4)

    defn conv_valid_no_stride_channels_last(inp, kernel) do
      Nx.conv(inp, kernel,
        padding: :valid,
        input_permutation: [:batch, :channels, :height, :width],
        output_permutation: [:batch, :channels, :height, :width]
      )
    end

    defn conv_same_stride_permuted(inp, kernel) do
      Nx.conv(inp, kernel,
        padding: :same,
        strides: [2, 1],
        input_permutation: [3, 2, 0, 1],
        kernel_permutation: [2, 0, 3, 1],
        output_permutation: [2, 3, 0, 1]
      )
    end

    defn conv_writes_default_output(inp, kernel) do
      Nx.conv(inp, kernel,
        padding: :valid,
        input_permutation: [:batch, :channels, :height, :width]
      )
    end

    defn batch_grouped_conv(inp, kernel) do
      Nx.conv(inp, kernel, batch_group_size: 2)
    end

    defn batch_grouped_conv_padding_dilated(inp, kernel) do
      Nx.conv(inp, kernel,
        batch_group_size: 4,
        padding: [{2, -1}, {1, 0}],
        input_dilation: [2, 1]
      )
    end

    test "computes a convolution with channels last" do
      img = Nx.iota({8, 12, 12, 3}, type: {:f, 32}, names: [:batch, :height, :width, :channels])
      kernel = Nx.iota({6, 3, 2, 2}, type: {:f, 32})

      lhs = conv_valid_no_stride_channels_last(img, kernel)

      rhs =
        Nx.conv(img, kernel,
          padding: :valid,
          input_permutation: [:batch, :channels, :height, :width],
          output_permutation: [:batch, :channels, :height, :width]
        )

      assert_all_close(lhs, rhs)
      assert %{names: [:batch, :height, :width, :channels], shape: {8, 11, 11, 6}} = lhs
    end

    test "computes a convolution with a permutation" do
      img = Nx.iota({12, 12, 3, 4}, type: {:f, 32})
      kernel = Nx.iota({3, 2, 32, 2}, type: {:f, 32})

      lhs = conv_same_stride_permuted(img, kernel)

      rhs =
        Nx.conv(img, kernel,
          padding: :same,
          strides: [2, 1],
          input_permutation: [3, 2, 0, 1],
          kernel_permutation: [2, 0, 3, 1],
          output_permutation: [2, 3, 0, 1]
        )

      assert_all_close(lhs, rhs)
      assert %{shape: {6, 12, 4, 32}} = lhs
    end

    test "computes a convolution with default channels first output despite input config" do
      img = Nx.iota({8, 12, 12, 3}, type: {:f, 32}, names: [:batch, :height, :width, :channels])
      kernel = Nx.iota({6, 3, 2, 2}, type: {:f, 32})

      lhs = conv_writes_default_output(img, kernel)

      rhs =
        Nx.conv(img, kernel,
          padding: :valid,
          input_permutation: [:batch, :channels, :height, :width]
        )

      assert_all_close(lhs, rhs)
      assert %{names: [:batch, :channels, :height, :width]} = lhs
    end

    @tag :unsupported_64_bit_op
    test "computes the convolution with valid padding, no stride" do
      img = Nx.iota({5, 1, 12, 12}, type: {:f, 64})
      kernel = Nx.iota({32, 1, 3, 3}, type: {:f, 64})

      lhs = conv_valid_no_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 1])
      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes the convolution with valid padding, {2, 2} stride" do
      img = Nx.iota({25, 1, 11, 8}, type: {:f, 64})
      kernel = Nx.iota({32, 1, 3, 3}, type: {:f, 64})

      lhs = conv_valid_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [2, 2], padding: :valid)
      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes the convolution with same padding, no stride" do
      img = Nx.iota({13, 3, 10, 6}, type: {:f, 64})
      kernel = Nx.iota({32, 3, 3, 3}, type: {:f, 64})

      lhs = conv_same_no_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 1], padding: :same)
      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes the convolution with same padding, stride" do
      img = Nx.iota({32, 1, 9, 9}, type: {:f, 64})
      kernel = Nx.iota({32, 1, 7, 7}, type: {:f, 64})

      lhs = conv_same_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [3, 3], padding: :same)
      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes the convolution with general padding, no stride" do
      img = Nx.iota({1, 1, 14, 14}, type: {:f, 64})
      kernel = Nx.iota({10, 1, 5, 5}, type: {:f, 64})

      lhs = conv_general_no_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 1], padding: [{-1, 2}, {3, -1}])

      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes the convolution with general padding, stride" do
      img = Nx.iota({2, 1, 12, 24}, type: {:f, 64})
      kernel = Nx.iota({2, 1, 6, 6}, type: {:f, 64})

      lhs = conv_general_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [2, 1], padding: [{2, 2}, {-2, 4}])

      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes a 3d convolution" do
      img = Nx.iota({3, 3, 5, 5, 5}, type: {:f, 64})
      kernel = Nx.iota({6, 3, 2, 2, 2}, type: {:f, 64})

      lhs = conv_3d(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 2, 1], padding: :same)

      assert_all_close(lhs, rhs)
    end

    test "computes a convolution with mixed types" do
      img = Nx.iota({3, 2, 10, 10})
      kernel = Nx.iota({6, 2, 4, 4}, type: {:f, 32})

      lhs = conv_valid_no_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 1], padding: :valid)

      assert_all_close(lhs, rhs)
    end

    @tag :unsupported_64_bit_op
    test "computes a dilated convolution" do
      img = Nx.iota({4, 3, 10, 10}, type: {:f, 64})
      kernel = Nx.iota({6, 3, 2, 2}, type: {:f, 64})

      lhs = dilated_conv(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 1], padding: :same, kernel_dilation: [1, 2])

      assert_all_close(lhs, rhs)
    end

    test "computes an input dilated convolution" do
      img = Nx.iota({4, 3, 10, 10}, type: {:f, 32})
      kernel = Nx.iota({6, 3, 2, 2}, type: {:f, 32})

      lhs = dilated_input_conv(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [1, 1], padding: :same, input_dilation: [2, 1])

      assert_all_close(lhs, rhs)
    end

    test "computes a conv with both dilations" do
      img = Nx.iota({4, 3, 15, 15}, type: {:f, 32})
      kernel = Nx.iota({6, 3, 3, 2}, type: {:f, 32})

      lhs = dilated_input_kernel_conv(img, kernel)

      rhs =
        Nx.conv(img, kernel,
          strides: [2, 1],
          padding: :same,
          input_dilation: [1, 2],
          kernel_dilation: [2, 1]
        )

      assert_all_close(lhs, rhs)
    end

    test "computes a grouped convolution with valid padding, no stride" do
      img = Nx.iota({4, 6, 10, 10}, type: {:f, 32})
      kernel = Nx.iota({6, 3, 2, 2}, type: {:f, 32})
      lhs = grouped_conv_valid_no_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: 1, padding: :valid, feature_group_size: 2)

      assert_all_close(lhs, rhs)
    end

    test "computes a grouped convolution with same padding, stride, 3-d" do
      img = Nx.iota({4, 8, 5, 7, 5}, type: {:f, 32})
      kernel = Nx.iota({4, 2, 2, 2, 1}, type: {:f, 32})

      lhs = grouped_conv_same_stride(img, kernel)
      rhs = Nx.conv(img, kernel, strides: [2, 1, 2], padding: :same, feature_group_size: 4)

      assert_all_close(lhs, rhs)
    end

    test "computes a batch grouped convolution" do
      img = Nx.iota({2, 4, 4, 4}, type: {:f, 32})
      kernel = Nx.iota({4, 4, 2, 2}, type: {:f, 32})

      lhs = batch_grouped_conv(img, kernel)
      rhs = Nx.conv(img, kernel, batch_group_size: 2)

      assert_all_close(lhs, rhs)
    end

    test "computes a batch grouped convolution with general padding, input dilation" do
      img = Nx.iota({8, 2, 4, 4}, type: {:f, 32})
      kernel = Nx.iota({4, 2, 2, 2}, type: {:f, 32})

      lhs = batch_grouped_conv_padding_dilated(img, kernel)

      rhs =
        Nx.conv(img, kernel,
          batch_group_size: 4,
          padding: [{2, -1}, {1, 0}],
          input_dilation: [2, 1]
        )

      assert_all_close(lhs, rhs)
    end
  end

  describe "transpose" do
    defn transpose(t), do: Nx.transpose(t)
    defn transpose_scalar(t), do: Nx.transpose(t, axes: [])
    defn transpose_perm1(t), do: Nx.transpose(t, axes: [2, 1, 0])
    defn transpose_perm2(t), do: Nx.transpose(t, axes: [2, 0, 1])
    defn transpose_perm3(t), do: Nx.transpose(t, axes: [0, 2, 1])

    test "transposes without axes" do
      assert_equal(transpose(Nx.tensor(1)), Nx.tensor(1))

      assert_equal(
        transpose(Nx.iota({2, 3, 4})),
        Nx.tensor([
          [[0, 12], [4, 16], [8, 20]],
          [[1, 13], [5, 17], [9, 21]],
          [[2, 14], [6, 18], [10, 22]],
          [[3, 15], [7, 19], [11, 23]]
        ])
      )
    end

    test "transposes with axes" do
      assert_equal(transpose_scalar(Nx.tensor(1)), Nx.tensor(1))

      assert_equal(
        transpose_perm1(Nx.iota({2, 3, 4})),
        Nx.tensor([
          [[0, 12], [4, 16], [8, 20]],
          [[1, 13], [5, 17], [9, 21]],
          [[2, 14], [6, 18], [10, 22]],
          [[3, 15], [7, 19], [11, 23]]
        ])
      )

      assert_equal(
        transpose_perm2(Nx.iota({2, 3, 4})),
        Nx.tensor([
          [[0, 4, 8], [12, 16, 20]],
          [[1, 5, 9], [13, 17, 21]],
          [[2, 6, 10], [14, 18, 22]],
          [[3, 7, 11], [15, 19, 23]]
        ])
      )

      assert_equal(
        transpose_perm3(Nx.iota({2, 3, 4})),
        Nx.tensor([
          [
            [0, 4, 8],
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11]
          ],
          [
            [12, 16, 20],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23]
          ]
        ])
      )
    end
  end

  describe "softmax" do
    defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

    test "computes softmax" do
      assert_all_close(
        softmax(Nx.tensor([1.0, 2.0, 3.0, 4.0])),
        Nx.tensor([
          0.03205860328008499,
          0.08714431874203257,
          0.23688281808991013,
          0.6439142598879722
        ])
      )
    end
  end

  describe "reshape" do
    defn reshape_with_shape(t), do: Nx.reshape(t, {2, 2})

    test "with shape" do
      assert_equal(reshape_with_shape(Nx.tensor([1, 2, 3, 4])), Nx.tensor([[1, 2], [3, 4]]))
    end

    defn reshape_with_tensor(t, shape), do: Nx.reshape(t, shape)

    test "with tensor" do
      assert_equal(
        reshape_with_tensor(Nx.tensor([1, 2, 3, 4]), Nx.tensor([[0, 0], [0, 0]])),
        Nx.tensor([[1, 2], [3, 4]])
      )

      assert_equal(
        reshape_with_tensor(Nx.tensor([1, 2, 3, 4]), Nx.tensor([[0], [0], [0], [0]])),
        Nx.tensor([[1], [2], [3], [4]])
      )
    end
  end

  describe "pad" do
    defn pad_scalar(t), do: Nx.pad(t, 0, [])
    defn pad_vector(t), do: Nx.pad(t, 0, [{1, 1, 0}])
    defn pad_matrix(t), do: Nx.pad(t, 0, [{1, 1, 0}, {1, 1, 0}])
    defn pad_tensor(t), do: Nx.pad(t, 0.0, [{1, 2, 0}, {1, 0, 0}, {0, 1, 0}])
    defn pad_vector_negative_value(t), do: Nx.pad(t, 0.0, [{-1, -1, 0}])
    defn pad_matrix_negative_value(t), do: Nx.pad(t, 0, [{0, 0, 0}, {-1, 1, 0}])
    defn pad_tensor_negative_value(t), do: Nx.pad(t, 0, [{-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}])

    test "with scalar" do
      assert_equal(pad_scalar(Nx.tensor(1)), Nx.tensor(1))
    end

    test "with vector" do
      assert_equal(pad_vector(Nx.tensor([1, 2, 3])), Nx.tensor([0, 1, 2, 3, 0]))
    end

    test "with matrix" do
      assert_equal(
        pad_matrix(Nx.tensor([[1, 2, 3], [4, 5, 6]])),
        Nx.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 0, 0, 0, 0]])
      )
    end

    test "with tensor" do
      result =
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

      assert_equal(pad_tensor(Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])), result)

      assert_equal(
        pad_tensor(Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])),
        result
      )
    end

    test "with negative value" do
      assert_equal(
        pad_vector_negative_value(Nx.tensor([1.0, 1.0, 2.0, 3.0, 0.0])),
        Nx.tensor([1.0, 2.0, 3.0])
      )

      assert_equal(
        pad_matrix_negative_value(Nx.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])),
        Nx.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
      )

      assert_equal(
        pad_tensor_negative_value(
          Nx.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 2, 0], [3, 4, 0], [0, 0, 0]],
            [[0, 0, 0], [5, 6, 0], [7, 8, 0], [0, 0, 0]]
          ])
        ),
        Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      )
    end
  end

  describe "broadcast" do
    defn broadcast_with_shape(t), do: Nx.broadcast(t, {2, 2})

    test "with shape" do
      assert_equal(broadcast_with_shape(Nx.tensor([1, 2])), Nx.tensor([[1, 2], [1, 2]]))
      assert_equal(broadcast_with_shape(Nx.tensor([[1, 2]])), Nx.tensor([[1, 2], [1, 2]]))
      assert_equal(broadcast_with_shape(Nx.tensor([[1], [2]])), Nx.tensor([[1, 1], [2, 2]]))
    end

    defn broadcast_with_tensor(t, shape), do: Nx.broadcast(t, shape)

    test "with tensor" do
      tensors = [
        {Nx.tensor([1, 2]), Nx.tensor([[[[0, 0]]]])},
        {Nx.tensor([[1, 2]]), Nx.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])}
      ]

      for {left, right} <- tensors do
        assert_equal(Nx.broadcast(left, right), broadcast_with_tensor(left, right))
      end
    end

    defn broadcast_with_axes_2(t), do: Nx.broadcast(t, {3, 2}, axes: [0])
    defn broadcast_with_axes_3(t), do: Nx.broadcast(t, {2, 3, 2}, axes: [1])

    test "with axes" do
      assert_equal(
        broadcast_with_axes_2(Nx.tensor([1, 2, 3])),
        Nx.tensor([[1, 1], [2, 2], [3, 3]])
      )

      assert_equal(
        broadcast_with_axes_3(Nx.tensor([1, 2, 3])),
        Nx.tensor([[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]])
      )
    end
  end

  describe "squeeze" do
    defn squeeze(t), do: Nx.squeeze(t)
    defn squeeze2(t), do: Nx.squeeze(t, axes: [0, 1])

    test "with scalar" do
      assert_equal(squeeze(Nx.tensor(1)), Nx.tensor(1))
    end

    test "with tensors" do
      assert_equal(squeeze(Nx.tensor([[1, 2, 3]])), Nx.tensor([1, 2, 3]))
      assert_equal(squeeze(Nx.tensor([[[[[1]]]]])), Nx.tensor(1))
      assert_equal(squeeze2(Nx.tensor([[[[[1]]]]])), Nx.tensor([[[1]]]))
    end
  end

  describe "random uniform" do
    defn random_uniform_fixed, do: Nx.random_uniform({30, 20})

    test "generates with shape" do
      t = random_uniform_fixed()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}

      for <<x::float-32-native <- Nx.to_binary(t)>> do
        assert x >= 0.0 and x < 1
      end
    end

    defn random_uniform_min_max_int, do: Nx.random_uniform({30, 20}, 5, 10)
    defn random_uniform_min_max_float, do: Nx.random_uniform({30, 20}, 5.0, 10.0)

    test "generates with min/max" do
      t = random_uniform_min_max_int()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:s, 64}

      for <<x::signed-64-native <- Nx.to_binary(t)>> do
        assert x >= 5 and x < 10
      end

      t = random_uniform_min_max_float()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}

      for <<x::float-32-native <- Nx.to_binary(t)>> do
        assert x >= 5.0 and x < 10.0
      end
    end

    defn random_uniform_u32, do: Nx.random_uniform({30, 20}, 5, 10, type: {:u, 32})
    defn random_uniform_f64, do: Nx.random_uniform({30, 20}, 5.0, 10.0, type: {:f, 64})

    @tag :unsupported_64_bit_op
    test "generates with type" do
      t = random_uniform_u32()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:u, 32}

      for <<x::unsigned-32-native <- Nx.to_binary(t)>> do
        assert x >= 5 and x < 10
      end

      t = random_uniform_f64()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 64}

      for <<x::float-64-native <- Nx.to_binary(t)>> do
        assert x >= 5.0 and x < 10.0
      end
    end

    defn random_uniform_tensor(min, max), do: Nx.random_uniform({30, 20}, min, max)

    test "generates with min/max tensor" do
      t = random_uniform_tensor(Nx.tensor(-100.0), Nx.tensor(100.0))
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}
    end
  end

  describe "random normal" do
    defn random_normal_fixed, do: Nx.random_normal({30, 20})

    test "generates with shape" do
      t = random_uniform_fixed()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}
    end

    defn random_normal_mu_sigma, do: Nx.random_normal({30, 20}, 5.0, 10.0)

    test "generates with mu/sigma" do
      t = random_normal_mu_sigma()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}
    end

    defn random_normal_f64, do: Nx.random_normal({30, 20}, 5.0, 10.0, type: {:f, 64})

    @tag :unsupported_64_bit_op
    test "generates with type" do
      t = random_normal_f64()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 64}
    end

    defn random_normal_tensor(mu, sigma), do: Nx.random_normal({30, 20}, mu, sigma)

    test "generates with tensor mu/sigma" do
      t = random_normal_tensor(Nx.tensor(1.0), Nx.tensor(1.0))
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}
    end
  end

  describe "iota" do
    defn iota_with_shape, do: Nx.iota({3, 4, 2, 3}, axis: 2)

    test "generates with shape" do
      assert_equal(iota_with_shape(), Nx.iota({3, 4, 2, 3}, axis: 2))
    end

    defn iota_with_type, do: Nx.iota({1, 2, 3}, axis: 1, type: {:f, 32})

    test "generates with type" do
      assert_equal(iota_with_type(), Nx.iota({1, 2, 3}, axis: 1, type: {:f, 32}))
    end

    defn iota_no_axis, do: Nx.iota({2, 2, 2})

    test "generates without axis" do
      assert_equal(iota_no_axis(), Nx.iota({2, 2, 2}))
    end

    defn iota_neg_axis, do: Nx.iota({2, 2, 2}, axis: -2)

    test "generates with negative axis" do
      assert_equal(iota_neg_axis(), Nx.iota({2, 2, 2}, axis: -2))
    end
  end

  describe "eye" do
    defn eye, do: Nx.eye(2)

    test "generates with shape" do
      assert_equal(eye(), Nx.tensor([[1, 0], [0, 1]]))
    end

    defn eye_with_type, do: Nx.eye(1, type: {:f, 32})

    test "generates with type" do
      assert_equal(eye_with_type(), Nx.tensor([[1]], type: {:f, 32}))
    end
  end

  describe "clip" do
    defn clip_both(value), do: Nx.clip(value, 2, 4)
    defn clip_mixed_types(value), do: Nx.clip(value, 2.0, 3)
    defn clip_with_tensor(value), do: Nx.clip(value, Nx.tensor(2.0), Nx.max(1.0, 3.0))

    test "works with both set" do
      assert_equal(
        clip_both(Nx.tensor([[1, 2, 3], [4, 5, 6]])),
        Nx.tensor([[2, 2, 3], [4, 4, 4]])
      )
    end

    test "works with mxied types" do
      assert_equal(
        clip_mixed_types(Nx.tensor([[1, 2, 3], [4, 5, 6]])),
        Nx.tensor([[2.0, 2.0, 3.0], [3.0, 3.0, 3.0]])
      )
    end

    test "works with tensor min/max" do
      assert_equal(
        clip_with_tensor(Nx.tensor([[1, 2, 3], [4, 5, 6]])),
        Nx.tensor([[2.0, 2.0, 3.0], [3.0, 3.0, 3.0]])
      )
    end

    test "works with floating point" do
      assert_equal(
        clip_both(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        Nx.tensor([[2.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
      )
    end
  end

  describe "slicing" do
    defn slice1(t), do: Nx.slice(t, [0, 6, 2], [2, 1, 3])

    defn slice1_dynamic(t), do: Nx.slice(t, [Nx.tensor(0), Nx.tensor(6), Nx.tensor(2)], [2, 1, 3])

    defn slice2(t), do: Nx.slice(t, [1, 4, 10], [1, 1, 10], strides: [1, 2, 3])

    defn slice2_dynamic(t),
      do: Nx.slice(t, [Nx.tensor(1), Nx.tensor(4), Nx.tensor(10)], [1, 1, 10], strides: [1, 2, 3])

    defn slice3(t), do: Nx.slice(t, [0, 4, 11], [2, 3, 9], strides: [2, 1, 3])

    defn slice3_dynamic(t),
      do: Nx.slice(t, [Nx.tensor(0), Nx.tensor(4), Nx.tensor(11)], [2, 3, 9], strides: [2, 1, 3])

    test "works without stride" do
      t = Nx.iota({900})
      t = Nx.reshape(t, {2, 15, 30})
      assert_equal(slice1(t), Nx.tensor([[[182, 183, 184]], [[632, 633, 634]]]))
      assert_equal(slice1_dynamic(t), Nx.tensor([[[182, 183, 184]], [[632, 633, 634]]]))
    end

    test "works with stride" do
      t = Nx.iota({900})
      t = Nx.reshape(t, {2, 15, 30})
      assert_equal(slice2(t), Nx.tensor([[[580, 583, 586, 589]]]))
      assert_equal(slice2_dynamic(t), Nx.tensor([[[580, 583, 586, 589]]]))

      assert_equal(
        slice3(t),
        Nx.tensor([
          [
            [131, 134, 137],
            [161, 164, 167],
            [191, 194, 197]
          ]
        ])
      )

      assert_equal(
        slice3_dynamic(t),
        Nx.tensor([
          [
            [131, 134, 137],
            [161, 164, 167],
            [191, 194, 197]
          ]
        ])
      )
    end
  end

  describe "put slice" do
    defn put_slice1(t1, t2), do: Nx.put_slice(t1, [2], t2)
    defn put_slice2(t1, t2), do: Nx.put_slice(t1, [1, 2], t2)
    defn put_slice3(t1, t2), do: Nx.put_slice(t1, [2, 2], t2)
    defn put_slice4(t1, t2), do: Nx.put_slice(t1, [Nx.tensor(0), Nx.tensor(2)], t2)

    test "works with one dimension" do
      assert_equal(
        put_slice1(Nx.tensor([0, 1, 2, 3, 4]), Nx.tensor([5, 6])),
        Nx.tensor([0, 1, 5, 6, 4])
      )
    end

    test "works with two dimensions" do
      assert_equal(
        put_slice2(Nx.tensor([[1, 2, 3], [4, 5, 6]]), Nx.tensor([[7, 8], [9, 10]])),
        Nx.tensor([[1, 7, 8], [4, 9, 10]])
      )
    end

    test "works with float types" do
      assert_equal(
        put_slice3(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        ),
        Nx.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
      )
    end

    test "works with mixed types" do
      assert_equal(
        put_slice4(Nx.tensor([[1, 2, 3], [4, 5, 6]]), Nx.tensor([[10.0, 11.0]])),
        Nx.tensor([[1.0, 10.0, 11.0], [4.0, 5.0, 6.0]])
      )
    end
  end

  describe "take" do
    defn take_axis_0(t, idx), do: Nx.take(t, idx)
    defn take_axis_1(t, idx), do: Nx.take(t, idx, axis: 1)

    test "1d indices" do
      assert_equal(
        take_axis_0(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 0, 1])),
        Nx.tensor([[3, 4], [1, 2], [3, 4]])
      )

      assert_equal(
        take_axis_1(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 0, 1])),
        Nx.tensor([[2, 1, 2], [4, 3, 4]])
      )
    end

    test "2d indices" do
      assert_equal(
        take_axis_1(
          Nx.tensor([[[1, 2], [11, 12]], [[101, 102], [111, 112]]]),
          Nx.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        ),
        Nx.tensor([
          [
            [
              [1, 2],
              [1, 2],
              [1, 2]
            ],
            [
              [11, 12],
              [11, 12],
              [11, 12]
            ],
            [
              [1, 2],
              [1, 2],
              [1, 2]
            ]
          ],
          [
            [
              [101, 102],
              [101, 102],
              [101, 102]
            ],
            [
              [111, 112],
              [111, 112],
              [111, 112]
            ],
            [
              [101, 102],
              [101, 102],
              [101, 102]
            ]
          ]
        ])
      )
    end
  end

  describe "gather" do
    defn gather(t, idx), do: Nx.gather(t, idx)

    test "1d result" do
      assert_equal(
        gather(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[1, 1], [0, 1], [1, 0]])),
        Nx.tensor([4, 2, 3])
      )

      assert_equal(
        gather(
          Nx.tensor([[[1, 2], [11, 12]], [[101, 102], [111, 112]]]),
          Nx.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]])
        ),
        Nx.tensor([1, 12, 112])
      )

      assert_equal(
        gather(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[-1, -1], [10, 11], [-1, 10]])),
        Nx.tensor([1, 4, 2])
      )
    end

    test "2d result" do
      assert_equal(
        gather(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 1]]])),
        Nx.tensor([[4, 1], [3, 2]])
      )
    end
  end

  describe "reverse" do
    defn reverse(t), do: Nx.reverse(t)
    defn reverse1(t), do: Nx.reverse(t, axes: [1])
    defn reverse2(t), do: Nx.reverse(t, axes: [0, 2])
    defn reverse3(t), do: Nx.reverse(t, axes: [1, 2, 4])

    test "works on all dims" do
      assert_equal(reverse(Nx.iota({10})), Nx.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
      assert_equal(reverse(Nx.iota({2, 4})), Nx.tensor([[7, 6, 5, 4], [3, 2, 1, 0]]))

      assert_equal(
        reverse(Nx.iota({3, 3, 3})),
        Nx.tensor([
          [[26, 25, 24], [23, 22, 21], [20, 19, 18]],
          [[17, 16, 15], [14, 13, 12], [11, 10, 9]],
          [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
        ])
      )

      assert_equal(
        reverse(Nx.iota({2, 1, 4, 2})),
        Nx.tensor([
          [[[15, 14], [13, 12], [11, 10], [9, 8]]],
          [[[7, 6], [5, 4], [3, 2], [1, 0]]]
        ])
      )
    end

    test "works with 1 dim" do
      assert_equal(reverse1(Nx.iota({2, 3})), Nx.tensor([[2, 1, 0], [5, 4, 3]]))
    end

    test "works with 2 dims" do
      assert_equal(
        reverse2(Nx.iota({2, 3, 4})),
        Nx.tensor([
          [
            [15, 14, 13, 12],
            [19, 18, 17, 16],
            [23, 22, 21, 20]
          ],
          [
            [3, 2, 1, 0],
            [7, 6, 5, 4],
            [11, 10, 9, 8]
          ]
        ])
      )
    end

    test "works with 3 dims" do
      assert_equal(
        reverse3(Nx.iota({2, 2, 1, 3, 4})),
        Nx.tensor([
          [
            [
              [
                [15, 14, 13, 12],
                [19, 18, 17, 16],
                [23, 22, 21, 20]
              ]
            ],
            [
              [
                [3, 2, 1, 0],
                [7, 6, 5, 4],
                [11, 10, 9, 8]
              ]
            ]
          ],
          [
            [
              [
                [39, 38, 37, 36],
                [43, 42, 41, 40],
                [47, 46, 45, 44]
              ]
            ],
            [
              [
                [27, 26, 25, 24],
                [31, 30, 29, 28],
                [35, 34, 33, 32]
              ]
            ]
          ]
        ])
      )
    end
  end

  describe "concatenate" do
    defn concatenate0(t1, t2, t3), do: Nx.concatenate([t1, t2, t3], axis: 0)
    defn concatenate1(t1, t2, t3), do: Nx.concatenate([t1, t2, t3], axis: 1)
    defn concatenate2(t1, t2, t3), do: Nx.concatenate([t1, t2, t3], axis: 2)
    defn concatenate1_inp(t1), do: Nx.concatenate([t1], axis: 2)
    defn concat_constants(), do: Nx.concatenate([Nx.tensor([1]), Nx.tensor([2])], axis: 0)

    test "works 0th axis" do
      t1 = Nx.iota({2, 2, 2})
      t2 = Nx.iota({1, 2, 2})
      t3 = Nx.iota({1, 2, 2})

      assert_equal(
        concatenate0(t1, t2, t3),
        Nx.tensor([
          [
            [0, 1],
            [2, 3]
          ],
          [
            [4, 5],
            [6, 7]
          ],
          [
            [0, 1],
            [2, 3]
          ],
          [
            [0, 1],
            [2, 3]
          ]
        ])
      )
    end

    test "works on 1st axis" do
      t1 = Nx.iota({1, 3, 2})
      t2 = Nx.iota({1, 1, 2})
      t3 = Nx.iota({1, 2, 2})

      assert_equal(
        concatenate1(t1, t2, t3),
        Nx.tensor([
          [
            [0, 1],
            [2, 3],
            [4, 5],
            [0, 1],
            [0, 1],
            [2, 3]
          ]
        ])
      )

      t1 = Nx.iota({2, 2, 2})
      t2 = Nx.add(t1, 10)
      t3 = Nx.add(t1, 20)

      assert_equal(
        concatenate1(t1, t2, t3),
        Nx.tensor([
          [
            [0, 1],
            [2, 3],
            [10, 11],
            [12, 13],
            [20, 21],
            [22, 23]
          ],
          [
            [4, 5],
            [6, 7],
            [14, 15],
            [16, 17],
            [24, 25],
            [26, 27]
          ]
        ])
      )
    end

    test "works on 2nd axis" do
      t1 = Nx.iota({2, 1, 4})
      t2 = Nx.iota({2, 1, 1})
      t3 = Nx.iota({2, 1, 3})

      assert_equal(
        concatenate2(t1, t2, t3),
        Nx.tensor([
          [
            [0, 1, 2, 3, 0, 0, 1, 2]
          ],
          [
            [4, 5, 6, 7, 1, 3, 4, 5]
          ]
        ])
      )
    end

    test "works with 1 input" do
      assert_equal(
        concatenate1_inp(Nx.iota({2, 1, 4})),
        Nx.tensor([
          [
            [0, 1, 2, 3]
          ],
          [
            [4, 5, 6, 7]
          ]
        ])
      )
    end

    test "works with mixed types" do
      t1 = Nx.iota({2, 2, 2}, type: {:f, 32})
      t2 = Nx.iota({1, 2, 2}, type: {:u, 8})
      t3 = Nx.iota({1, 2, 2}, type: {:bf, 16})

      assert_equal(
        concatenate0(t1, t2, t3),
        Nx.tensor(
          [
            [
              [0.0, 1.0],
              [2.0, 3.0]
            ],
            [
              [4.0, 5.0],
              [6.0, 7.0]
            ],
            [
              [0.0, 1.0],
              [2.0, 3.0]
            ],
            [
              [0.0, 1.0],
              [2.0, 3.0]
            ]
          ],
          type: {:f, 32}
        )
      )
    end

    test "works with constants" do
      assert_equal(concat_constants(), Nx.tensor([1, 2]))
    end

    test "works with mixed backends" do
      opts = [[backend: Nx.BinaryBackend], []]

      for opts1 <- opts, opts2 <- opts do
        t1 = Nx.tensor([1, 2], opts1)
        t2 = Nx.tensor([3, 4], opts2)
        t3 = Nx.tensor([5, 6], opts1)

        assert_all_close(Nx.tensor([1, 2, 3, 4, 5, 6]), Nx.concatenate([t1, t2, t3]))
      end
    end
  end

  describe "decompositions" do
    defn ts(a, b, opts \\ []), do: Nx.LinAlg.triangular_solve(a, b, opts)

    test "triangular_solve" do
      a = Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      b = Nx.tensor([4, 2, 4, 2])
      assert_all_close(Nx.dot(a, ts(a, b)), b)

      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1, 2, 3], [2, 2, 4], [2, 0, 1]])
      assert_all_close(Nx.dot(a, ts(a, b)), b)

      upper = Nx.transpose(a)
      assert_all_close(Nx.dot(upper, ts(upper, b, lower: false)), b)
      assert_all_close(Nx.dot(ts(upper, b, left_side: false, lower: false), upper), b)
      assert_all_close(Nx.dot(Nx.transpose(a), ts(a, b, transform_a: :transpose)), b)
    end

    defn qr(t), do: Nx.LinAlg.qr(t)
    defn qr_complete(t), do: Nx.LinAlg.qr(t, mode: :complete)

    test "qr" do
      input = Nx.iota({3, 2})
      output = Nx.as_type(input, {:f, 32})

      assert {q, r} = qr(input)
      assert q.shape == {3, 2}
      assert r.shape == {2, 2}
      assert_all_close(Nx.dot(q, r), output)

      assert {q, r} = qr_complete(Nx.iota({3, 2}))
      assert q.shape == {3, 3}
      assert r.shape == {3, 2}
      assert_all_close(Nx.dot(q, r), output)
    end

    defn svd(t), do: Nx.LinAlg.svd(t)

    test "svd" do
      input = Nx.iota({3, 3})
      output = Nx.as_type(input, {:f, 32})

      assert {u, s, vt} = svd(input)
      assert u.shape == {3, 3}
      assert s.shape == {3}
      assert vt.shape == {3, 3}
      s_full = Nx.multiply(s, Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

      assert_all_close(u |> Nx.dot(s_full) |> Nx.dot(vt), output,
        atol: 1.0e-5,
        rtol: 1.0e-2
      )
    end

    test "svd (tall matrix)" do
      input = Nx.tensor([[2, 0], [0, 1], [0, 0]])
      output = Nx.as_type(input, {:f, 32})

      assert {u, s, vt} = svd(input)
      assert u.shape == {3, 3}
      assert s.shape == {2}
      assert vt.shape == {2, 2}
      s_full = Nx.multiply(s, Nx.tensor([[1, 0], [0, 1], [0, 0]]))

      assert_all_close(u |> Nx.dot(s_full) |> Nx.dot(vt), output,
        atol: 1.0e-5,
        rtol: 1.0e-2
      )
    end

    test "svd (wide matrix)" do
      input = Nx.tensor([[2, 0, 0], [0, 1, 0]])
      output = Nx.as_type(input, {:f, 32})

      assert {u, s, vt} = svd(input)
      assert u.shape == {2, 2}
      assert s.shape == {2}
      assert vt.shape == {3, 3}
      s_full = Nx.multiply(Nx.reshape(s, {2, 1}), Nx.tensor([[1, 0, 0], [0, 1, 0]]))

      assert_all_close(u |> Nx.dot(s_full) |> Nx.dot(Nx.transpose(vt)), output,
        atol: 1.0e-5,
        rtol: 1.0e-2
      )
    end
  end

  describe "sort" do
    defn sort0(t), do: Nx.sort(t, axis: 0)
    defn sort1(t), do: Nx.sort(t, axis: 1)
    defn sort1_asc(t), do: Nx.sort(t, axis: 1, direction: :asc)
    defn sort2(t), do: Nx.sort(t, axis: 2)

    test "sorts a 1d tensor" do
      assert_equal(sort0(Nx.tensor([0, 5, 2, 1, 3, 4])), Nx.tensor([0, 1, 2, 3, 4, 5]))
    end

    test "sorts a 2d tensor along the 0th axis" do
      assert_equal(sort0(Nx.tensor([[3, 1, 7], [2, 5, 4]])), Nx.tensor([[2, 1, 4], [3, 5, 7]]))
    end

    test "sorts a 2d tensor along the 1st axis" do
      assert_equal(sort1(Nx.tensor([[3, 1, 7], [2, 5, 4]])), Nx.tensor([[1, 3, 7], [2, 4, 5]]))
    end

    test "sorts a 2d tensor along the 1st axis ascending" do
      assert_equal(
        sort1_asc(Nx.tensor([[3, 1, 7], [2, 5, 4]])),
        Nx.tensor([[1, 3, 7], [2, 4, 5]])
      )
    end

    test "sorts a 3d tensor along the 2nd axis" do
      assert_equal(
        sort2(Nx.tensor([[[4, 5, 2], [2, 5, 3], [5, 0, 2]], [[1, 9, 8], [2, 1, 3], [2, 1, 4]]])),
        Nx.tensor([
          [
            [2, 4, 5],
            [2, 3, 5],
            [0, 2, 5]
          ],
          [
            [1, 8, 9],
            [1, 2, 3],
            [1, 2, 4]
          ]
        ])
      )
    end
  end

  describe "top_k" do
    defn top_1(t), do: Nx.top_k(t, k: 1)

    test "returns top 1 values and indices" do
      a = Nx.iota({5})
      assert_equal(top_1(a), {Nx.tensor([4]), Nx.tensor([4])})

      a = Nx.iota({5, 5}, axis: 1)
      assert_equal(top_1(a), {Nx.broadcast(4, {5, 1}), Nx.broadcast(4, {5, 1})})

      a = Nx.iota({5, 5, 5}, axis: 2)
      assert_equal(top_1(a), {Nx.broadcast(4, {5, 5, 1}), Nx.broadcast(4, {5, 5, 1})})
    end
  end

  describe "argsort" do
    defn argsort0(t), do: Nx.argsort(t, axis: 0)
    defn argsort1(t), do: Nx.argsort(t, axis: 1)
    defn argsort1_asc(t), do: Nx.argsort(t, axis: 1, direction: :asc)
    defn argsort2(t), do: Nx.argsort(t, axis: 2)

    test "sorts a 1d tensor and returns its indices" do
      assert_equal(argsort0(Nx.tensor([0, 5, 2, 1, 3, 4])), Nx.tensor([0, 3, 2, 4, 5, 1]))
    end

    test "sorts a 2d tensor along the 0th axis and returns its indices" do
      assert_equal(argsort0(Nx.tensor([[3, 1, 7], [2, 5, 4]])), Nx.tensor([[1, 0, 1], [0, 1, 0]]))
    end

    test "sorts a 2d tensor along the 1st axis and returns its indices" do
      assert_equal(argsort1(Nx.tensor([[3, 1, 7], [2, 5, 4]])), Nx.tensor([[1, 0, 2], [0, 2, 1]]))
    end

    test "sorts a 2d tensor along the 1st axis ascending and returns its indices" do
      assert_equal(
        argsort1_asc(Nx.tensor([[3, 1, 7], [2, 5, 4]])),
        Nx.tensor([[1, 0, 2], [0, 2, 1]])
      )
    end

    test "sorts a 3d tensor along the 2nd axis and returns its indices" do
      assert_equal(
        argsort2(
          Nx.tensor([[[4, 5, 2], [2, 5, 3], [5, 0, 2]], [[1, 9, 8], [2, 1, 3], [2, 1, 4]]])
        ),
        Nx.tensor([
          [
            [2, 0, 1],
            [0, 2, 1],
            [1, 2, 0]
          ],
          [
            [0, 2, 1],
            [1, 0, 2],
            [1, 0, 2]
          ]
        ])
      )
    end

    test "sorts a floating-point tensor and returns its indices" do
      assert_equal(
        argsort0(Nx.tensor([42.0, 23.0, 16.0, 15.0, 8.0, 4.0])),
        Nx.tensor([5, 4, 3, 2, 1, 0])
      )
    end
  end

  describe "optional" do
    defn determinant(t), do: Nx.LinAlg.determinant(t)

    test "determinant" do
      two_by_two = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      assert_equal(determinant(two_by_two), Nx.tensor(-2.0))
    end

    defn double_determinant(a, b), do: Nx.LinAlg.determinant(a) * Nx.LinAlg.determinant(b)

    test "multiple determinant" do
      from_one = Nx.tensor([[1, 2], [3, 4]])
      from_ten = Nx.tensor([[10, 20], [30, 40]])
      assert_equal(double_determinant(from_one, from_ten), Nx.tensor(400.0))
    end
  end

  describe "cholesky" do
    defn cholesky(t), do: Nx.LinAlg.cholesky(t)

    test "works on 2x2 matrix" do
      lhs = cholesky(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]))
      rhs = Nx.tensor([[4.47213595499958, 0.0], [3.93547964039963, 0.7155417527999305]])
      assert_all_close(lhs, rhs)

      lhs = cholesky(Nx.tensor([[1, 2], [2, 5]]))
      rhs = Nx.tensor([[1.0, 0.0], [2.0, 1.0]])
      assert_all_close(lhs, rhs)
    end

    test "works on a 4x4 matrix" do
      lhs =
        cholesky(
          Nx.tensor([
            [6.0, 3.0, 4.0, 8.0],
            [3.0, 6.0, 5.0, 1.0],
            [4.0, 5.0, 10.0, 7.0],
            [8.0, 1.0, 7.0, 25.0]
          ])
        )

      rhs =
        Nx.tensor([
          [2.449489742783178, 0.0, 0.0, 0.0],
          [1.2247448713915892, 2.1213203435596424, 0.0, 0.0],
          [1.6329931618554523, 1.414213562373095, 2.309401076758503, 0.0],
          [3.2659863237109046, -1.4142135623730956, 1.5877132402714704, 3.1324910215354165]
        ])

      assert_all_close(lhs, rhs)
    end

    test "works on a 50x50 matrix" do
      tensor = Nx.random_normal({50, 50})
      tensor = Nx.dot(tensor, Nx.transpose(tensor))
      tensor = Nx.add(tensor, Nx.multiply(50, Nx.eye(Nx.shape(tensor))))

      l = cholesky(tensor)
      assert_all_close(Nx.dot(l, Nx.transpose(l)), tensor, atol: 1.0e-4, rtol: 1.0e-2)
    end
  end

  describe "bfloat16" do
    defn add(t1, t2), do: t1 + t2

    test "accepts bfloat16 input" do
      lhs = Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16})
      rhs = Nx.tensor([4.0, 5.0, 6.0], type: {:bf, 16})
      assert_equal(add(lhs, rhs), Nx.tensor([5.0, 7.0, 9.0], type: {:bf, 16}))
    end
  end

  describe "precision" do
    defn precision(t1, t2), do: Nx.dot(t1, t2)

    test "raises on bad precision" do
      assert_raise ArgumentError,
                   "expected precision configuration to be one of" <>
                     " :default, :high, or :highest, got: :bad",
                   fn ->
                     EXLA.jit(&precision/2, precision: :bad).(
                       Nx.tensor([1, 2, 3], type: {:bf, 16}),
                       Nx.tensor([1, 2, 3], type: {:bf, 16})
                     )
                   end
    end

    test "succeeds on good precision" do
      assert_equal(
        EXLA.jit(&precision/2, precision: :high).(
          Nx.tensor([1, 2, 3], type: {:bf, 16}),
          Nx.tensor([1, 2, 3], type: {:bf, 16})
        ),
        Nx.tensor(14, type: {:bf, 16})
      )
    end
  end

  describe "take_along_axis/3" do
    defn take_along_axis(t, idx, opts \\ [axis: 0]), do: Nx.take_along_axis(t, idx, opts)

    defn sort_with_take_along_axis(t, opts \\ []) do
      idx = Nx.argsort(t, opts)
      Nx.take_along_axis(t, idx, axis: opts[:axis])
    end

    test "works for {3, 2, 2} tensor along axis 2" do
      t = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
      i = Nx.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]])

      assert_equal(
        take_along_axis(t, i, axis: 2),
        Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
      )
    end

    test "works for {3, 2, 2} tensor growing along axis = 2" do
      t = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

      i =
        Nx.tensor([
          [[0, 1, 1, 0], [0, 1, 1, 0]],
          [[0, 1, 1, 0], [0, 1, 1, 0]],
          [[0, 1, 1, 0], [0, 1, 1, 0]]
        ])

      assert_equal(
        take_along_axis(t, i, axis: 2),
        Nx.tensor([
          [[1, 2, 2, 1], [3, 4, 4, 3]],
          [[5, 6, 6, 5], [7, 8, 8, 7]],
          [[9, 10, 10, 9], [11, 12, 12, 11]]
        ])
      )
    end

    test "works for {3, 2, 2} tensor growing along axis = 0" do
      t = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

      i =
        Nx.tensor([
          [
            [0, 0],
            [0, 0]
          ],
          [
            [1, 1],
            [1, 1]
          ],
          [
            [2, 2],
            [2, 2]
          ],
          [
            [2, 1],
            [1, 0]
          ]
        ])

      assert_equal(
        take_along_axis(t, i),
        Nx.tensor([
          [[1, 2], [3, 4]],
          [[5, 6], [7, 8]],
          [[9, 10], [11, 12]],
          [[9, 6], [7, 4]]
        ])
      )
    end

    test "uses argsort indices properly" do
      t = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

      assert_equal(
        sort_with_take_along_axis(t, axis: 1, direction: :desc),
        Nx.sort(t, axis: 1, direction: :desc)
      )
    end
  end

  defn while_in_cond(i) do
    cond do
      i < 5 ->
        while {i}, i < 5 do
          {i + 1}
        end

      true ->
        {i}
    end
  end

  test "computes while inside cond" do
    assert {i} = while_in_cond(0)
    assert_equal(i, Nx.tensor(5))
  end
end
