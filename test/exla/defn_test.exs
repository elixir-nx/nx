defmodule Exla.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler Exla

  describe "scalar" do
    defn just_two_int, do: 2
    defn just_two_float, do: 2.0

    test "returns the tensor for the scalar" do
      assert just_two_int() == Nx.tensor(2)
      assert just_two_float() == Nx.tensor(2.0)
    end
  end

  describe "tuples" do
    defn two_constant_tuples, do: {-1, 1.0}
    defn three_constant_tuples, do: {1, 2.0, 3}

    test "returns tuples with constants" do
      assert two_constant_tuples() == {Nx.tensor(-1), Nx.tensor(1.0)}
      assert three_constant_tuples() == {Nx.tensor(1), Nx.tensor(2.0), Nx.tensor(3)}
    end

    defn add_subtract_tuple(a, b), do: {a + b, a - b}

    test "returns tuples with operation results" do
      assert add_subtract_tuple(2, 3) == {Nx.tensor(5), Nx.tensor(-1)}

      assert add_subtract_tuple(Nx.tensor([-1, 0, 1]), 10) ==
               {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
    end

    defn pattern_tuple({a, b}), do: a + b

    test "matches on tuples" do
      assert pattern_tuple({2, 3}) == Nx.tensor(5)

      assert pattern_tuple({Nx.tensor([1, 2]), Nx.tensor([[3], [4]])}) ==
               Nx.tensor([[4, 5], [5, 6]])
    end

    defn calls_pattern_tuple(a, b), do: pattern_tuple({a, b})

    test "matches on inlined tuples" do
      assert calls_pattern_tuple(2, 3) == Nx.tensor(5)

      assert calls_pattern_tuple(Nx.tensor([1, 2]), Nx.tensor([[3], [4]])) ==
               Nx.tensor([[4, 5], [5, 6]])
    end
  end

  describe "tensor constants" do
    @two 2
    defn add_two_attribute(t), do: t + @two

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn add_2x2_attribute(t), do: t + @two_per_two

    test "expands module attributes to scalars" do
      assert add_two_attribute(1) == Nx.tensor(3)
      assert add_two_attribute(Nx.tensor([1, 2, 3])) == Nx.tensor([3, 4, 5])
    end

    test "expands module attributes to tensors" do
      assert add_2x2_attribute(1) == Nx.tensor([[2, 3], [4, 5]])
      assert add_2x2_attribute(Nx.tensor([1, 2])) == Nx.tensor([[2, 4], [4, 6]])
    end
  end

  describe "+/2" do
    defn add_two(a, b), do: a + b
    @defn_compiler Nx.Defn
    defn add_two_nx(a, b), do: a + b

    test "same shape and type" do
      assert add_two(1.0, 2.0) == Nx.tensor(3.0)
      assert add_two(1, 2) == Nx.tensor(3)

      assert add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4])) == Nx.tensor([4, 6])
      assert add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])) == Nx.tensor([4.0, 6.0])
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
        compare_tensors!(add_two(left, right), add_two_nx(left, right))
        compare_tensors!(add_two(right, left), add_two_nx(right, left))
      end
    end

    defn add_two_int_int, do: 1 + 2
    @defn_compiler Nx.Defn
    defn add_two_int_int_nx, do: 1 + 2

    defn add_two_int_float, do: 1 + 2.0
    @defn_compiler Nx.Defn
    defn add_two_int_float_nx, do: 1 + 2.0

    defn add_two_float_int, do: 1.0 + 2
    @defn_compiler Nx.Defn
    defn add_two_float_int_nx, do: 1.0 + 2

    defn add_two_float_float, do: 1.0 + 2
    @defn_compiler Nx.Defn
    defn add_two_float_float_nx, do: 1.0 + 2

    defn add_two_int(t), do: t + 2
    @defn_compiler Nx.Defn
    defn add_two_int_nx(t), do: t + 2

    defn add_two_float(t), do: t + 2.0
    @defn_compiler Nx.Defn
    defn add_two_float_nx(t), do: t + 2.0

    test "constants" do
      assert add_two_int_int() == add_two_int_int_nx()
      assert add_two_int_float() == add_two_int_float_nx()
      assert add_two_float_int() == add_two_float_int_nx()
      assert add_two_float_float() == add_two_float_float_nx()

      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:f, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
      ]

      for t <- tensors do
        compare_tensors!(add_two_int(t), add_two_int_nx(t))
        compare_tensors!(add_two_float(t), add_two_float_nx(t))
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
        compare_tensors!(add_two(left, right), add_two_nx(left, right))
        compare_tensors!(add_two(right, left), add_two_nx(right, left))
      end
    end

    test "broadcast error" do
      assert_raise RuntimeError, ~r"Binary op add with incompatible shapes", fn ->
        add_two(Nx.tensor([1, 2]), Nx.tensor([1, 2, 3]))
      end
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
        compare_tensors!(divide_two(left, right), Nx.divide(left, right))
        compare_tensors!(divide_two(right, left), Nx.divide(right, left))
      end
    end

    defn divide_two_int(t), do: t / 2
    defn divide_two_float(t), do: t / 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:f, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
      ]

      for t <- tensors do
        compare_tensors!(divide_two_int(t), Nx.divide(t, 2))
        compare_tensors!(divide_two_float(t), Nx.divide(t, 2.0))
      end
    end
  end

  describe "remainder" do
    defn remainder(a, b), do: Nx.remainder(a, b)

    test "integers" do
      left = Nx.tensor([-1023, 1023])
      right = Nx.tensor([[-4], [4]])
      assert Nx.shape(remainder(left, right)) == {2, 2}
      compare_tensors!(remainder(left, right), Nx.remainder(left, right))
    end

    test "floats" do
      left = Nx.tensor([-8.3, -8.4, -8.5, 8.3, 8.4, 8.5])
      right = Nx.tensor([[-4.2], [-4.1], [-4.0], [4.0], [4.1], [4.2]])
      assert Nx.shape(remainder(left, right)) == {6, 6}
      compare_tensors!(remainder(left, right), Nx.remainder(left, right))
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
        compare_tensors!(subtract_two(left, right), Nx.subtract(left, right))
        compare_tensors!(subtract_two(right, left), Nx.subtract(right, left))
      end
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      for {left, right} <- @tensors do
        compare_tensors!(multiply_two(left, right), Nx.multiply(left, right))
        compare_tensors!(multiply_two(right, left), Nx.multiply(right, left))
      end
    end

    defn unary_minus(a), do: -a

    test "negate" do
      for t <- [
            Nx.tensor([-1, 0, 1], type: {:u, 8}),
            Nx.tensor([-1, 0, 1]),
            Nx.tensor([-1.0, 1.0])
          ] do
        assert unary_minus(t) == Nx.negate(t)
      end
    end

    defn max_two(a, b), do: max(a, b)

    test "max" do
      for {left, right} <- @tensors do
        compare_tensors!(max_two(left, right), Nx.max(left, right))
        compare_tensors!(max_two(right, left), Nx.max(right, left))
      end
    end

    defn min_two(a, b), do: min(a, b)

    test "min" do
      for {left, right} <- @tensors do
        compare_tensors!(min_two(left, right), Nx.min(left, right))
        compare_tensors!(min_two(right, left), Nx.min(right, left))
      end
    end

    defn power_two(a, b), do: Nx.power(a, b)

    test "power" do
      for {left, right} <- @tensors do
        compare_tensors!(power_two(left, right), Nx.power(left, right))
        compare_tensors!(power_two(right, left), Nx.power(right, left))
      end
    end

    defn arctan2_two(a, b), do: Nx.arctan2(a, b)

    test "arctan2" do
      <<neg_zero::float>> = <<0x8000000000000000::64>>
      left = Nx.tensor([-1.0, neg_zero, 0.0, 1.0])
      right = Nx.tensor([[-1.0], [neg_zero], [0.0], [1.0]])

      compare_tensors!(arctan2_two(left, right), Nx.arctan2(left, right))
      compare_tensors!(arctan2_two(right, left), Nx.arctan2(right, left))
    end
  end

  describe "element-wise bitwise operators" do
    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[-2], [-1], [0], [1], [2]])

    defn bitwise_and(a, b), do: a &&& b

    test "bitwise_and" do
      assert Nx.shape(bitwise_and(@left, @right)) == {5, 5}
      assert bitwise_and(@left, @right) == Nx.bitwise_and(@left, @right)
    end

    defn bitwise_or(a, b), do: a ||| b

    test "bitwise_or" do
      assert Nx.shape(bitwise_or(@left, @right)) == {5, 5}
      assert bitwise_or(@left, @right) == Nx.bitwise_or(@left, @right)
    end

    defn bitwise_xor(a, b), do: a ^^^ b

    test "bitwise_xor" do
      assert Nx.shape(bitwise_xor(@left, @right)) == {5, 5}
      assert bitwise_xor(@left, @right) == Nx.bitwise_xor(@left, @right)
    end

    defn bitwise_not(a), do: ~~~a

    test "bitwise_not" do
      assert Nx.shape(bitwise_not(@left)) == {5}
      assert bitwise_not(@left) == Nx.bitwise_not(@left)
    end

    defn bitwise_pc(a), do: Nx.population_count(a)

    test "population_count" do
      assert Nx.shape(bitwise_pc(@left)) == {5}
      assert bitwise_pc(@left) == Nx.population_count(@left)
    end

    defn bitwise_clz(a), do: Nx.count_leading_zeros(a)

    test "count_leading_zeros" do
      assert Nx.shape(bitwise_clz(@left)) == {5}
      assert bitwise_clz(@left) == Nx.count_leading_zeros(@left)
    end

    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[0], [1], [2], [3], [4]])

    defn left_shift(a, b), do: a <<< b

    test "left_shift" do
      assert Nx.shape(left_shift(@left, @right)) == {5, 5}
      assert left_shift(@left, @right) == Nx.left_shift(@left, @right)
    end

    @left_signed Nx.tensor([-128, -127, -2, -1, 0, 1, 2, 126, 127], type: {:s, 8})
    @right_signed Nx.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8]], type: {:s, 8})

    @left_unsigned Nx.tensor([0, 1, 2, 253, 254, 255], type: {:u, 8})
    @right_unsigned Nx.tensor([[0], [1], [2], [3], [4], [5]], type: {:u, 8})

    defn right_shift(a, b), do: a >>> b

    test "right_shift" do
      assert Nx.shape(right_shift(@left_signed, @right_signed)) == {9, 9}

      assert right_shift(@left_signed, @right_signed) ==
               Nx.right_shift(@left_signed, @right_signed)

      assert Nx.shape(right_shift(@left_unsigned, @right_unsigned)) == {6, 6}

      assert right_shift(@left_unsigned, @right_unsigned) ==
               Nx.right_shift(@left_unsigned, @right_unsigned)
    end
  end

  describe "exp" do
    defn exp(t), do: Nx.exp(t)

    test "computes the exp across types" do
      assert Nx.tensor([1, 2, 3]) |> exp() ==
               Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])

      assert Nx.tensor([1, 2, 3], type: {:s, 8}) |> exp() ==
               Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})

      assert Nx.tensor([1, 2, 3], type: {:u, 8}) |> exp() ==
               Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})

      assert Nx.tensor([1.0, 2.0, 3.0]) |> exp() ==
               Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])

      assert Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> exp() ==
               Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668], type: {:f, 32})
    end

    defn exp_int(), do: Nx.exp(1)
    defn exp_float(), do: Nx.exp(1.0)

    test "constants" do
      assert exp_int() == Nx.tensor(2.718281828459045)
      assert exp_float() == Nx.tensor(2.718281828459045)
    end
  end

  describe "equal" do
    defn equal(a, b), do: Nx.equal(a, b)

    test "computes equality of scalars" do
      assert equal(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(0, type: {:u, 8})
    end

    test "computes equality with broadcasting" do
      assert equal(Nx.tensor(1), Nx.tensor([1, 2, 3])) == Nx.tensor([1, 0, 0], type: {:u, 8})
    end

    test "computes equality with mixed types" do
      assert equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([1, 1, 1], type: {:u, 8})
    end
  end

  describe "not equal" do
    defn not_equal(a, b), do: Nx.not_equal(a, b)

    test "computes equality of scalars" do
      assert not_equal(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(1, type: {:u, 8})
    end

    test "computes equality with broadcasting" do
      assert not_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])) == Nx.tensor([0, 1, 1], type: {:u, 8})
    end

    test "computes equality with mixed types" do
      assert not_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([0, 0, 0], type: {:u, 8})
    end
  end

  describe "less" do
    defn less(a, b), do: Nx.less(a, b)

    test "compares scalars" do
      assert less(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(1, type: {:u, 8})
    end

    test "compares with broadcasting" do
      assert less(Nx.tensor(1), Nx.tensor([1, 2, 3])) == Nx.tensor([0, 1, 1], type: {:u, 8})
    end

    test "compares with mixed types" do
      assert less(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([0, 0, 0], type: {:u, 8})
    end
  end

  describe "greater" do
    defn greater(a, b), do: Nx.greater(a, b)

    test "compares scalars" do
      assert greater(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(0, type: {:u, 8})
    end

    test "compares with broadcasting" do
      assert greater(Nx.tensor(1), Nx.tensor([1, 2, 3])) == Nx.tensor([0, 0, 0], type: {:u, 8})
    end

    test "compares with mixed types" do
      assert greater(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([0, 0, 0], type: {:u, 8})
    end
  end

  describe "less equal" do
    defn less_equal(a, b), do: Nx.less_equal(a, b)

    test "compares scalars" do
      assert less_equal(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(1, type: {:u, 8})
    end

    test "compares with broadcasting" do
      assert less_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])) == Nx.tensor([1, 1, 1], type: {:u, 8})
    end

    test "compares with mixed types" do
      assert less_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([1, 1, 1], type: {:u, 8})
    end
  end

  describe "greater equal" do
    defn greater_equal(a, b), do: Nx.greater_equal(a, b)

    test "compares scalars" do
      assert greater_equal(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(0, type: {:u, 8})
    end

    test "compares with broadcasting" do
      assert greater_equal(Nx.tensor(1), Nx.tensor([1, 2, 3])) == Nx.tensor([1, 0, 0], type: {:u, 8})
    end

    test "compares with mixed types" do
      assert greater_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([1, 1, 1], type: {:u, 8})
    end
  end

  describe "select" do
    defn select(pred, x, y), do: Nx.select(pred, x, y)

    test "selects one or the other with a scalar" do
      assert select(Nx.tensor(1), Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])) == Nx.tensor([1, 2, 3])
    end

    test "selects with broadcasting" do
      assert select(Nx.tensor([1, 0, 1, 0, 1]), Nx.tensor([10]), Nx.tensor([1, 2, 3, 4, 5])) == Nx.tensor([10, 2, 10, 4, 10])
    end
  end

  describe "unary float ops" do
    @int_tensor Nx.tensor([1, 2, 3])
    @float_tensor Nx.tensor([1.0, 2.0, 3.0])

    for fun <- [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] do
      exla_fun = :"unary_#{fun}"
      nx_fun = :"unary_#{fun}_nx"
      defn unquote(exla_fun)(t), do: Nx.unquote(fun)(t)
      @defn_compiler Nx.Defn
      defn unquote(nx_fun)(t), do: Nx.unquote(fun)(t)

      test "#{fun}" do
        compare_tensors!(unquote(exla_fun)(@float_tensor), unquote(nx_fun)(@float_tensor))
        compare_tensors!(unquote(exla_fun)(@int_tensor), unquote(nx_fun)(@int_tensor))
      end
    end
  end

  describe "unary round+sign ops" do
    @uint_tensor Nx.tensor([0, 1, 2], type: {:u, 8})
    @sint_tensor Nx.tensor([-2, -1, 0, 1, 2])
    @float_tensor Nx.tensor([-1.5, 0.5, -0.0, 0.0, 0.5, 1.5])

    funs = [:abs, :sign, :floor, :ceil, :round]

    for fun <- funs do
      exla_fun = :"unary_#{fun}"
      nx_fun = :"unary_#{fun}_nx"
      defn unquote(exla_fun)(t), do: Nx.unquote(fun)(t)
      @defn_compiler Nx.Defn
      defn unquote(nx_fun)(t), do: Nx.unquote(fun)(t)

      test "#{fun}" do
        compare_tensors!(unquote(exla_fun)(@uint_tensor), unquote(nx_fun)(@uint_tensor))
        compare_tensors!(unquote(exla_fun)(@sint_tensor), unquote(nx_fun)(@sint_tensor))
        compare_tensors!(unquote(exla_fun)(@float_tensor), unquote(nx_fun)(@float_tensor))
      end
    end
  end

  describe "sum" do
    defn sum(t), do: Nx.sum(t)

    test "computes the sum across types" do
      assert Nx.tensor([1, 2, 3]) |> sum() == Nx.tensor(6)
      assert Nx.tensor([1, 2, 3], type: {:s, 8}) |> sum() == Nx.tensor(6, type: {:s, 8})
      assert Nx.tensor([1, 2, 3], type: {:u, 8}) |> sum() == Nx.tensor(6, type: {:u, 8})
      assert Nx.tensor([1.0, 2.0, 3.0]) |> sum() == Nx.tensor(6.0)
      assert Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> sum() == Nx.tensor(6, type: {:f, 32})
    end

    defn sum_pos_axis(t), do: Nx.sum(t, axes: [1])
    defn sum_neg_axis(t), do: Nx.sum(t, axes: [-3])
    defn sum_pos_neg_axis(t), do: Nx.sum(t, axes: [1, -3])

    test "computes the sum on a given axis" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      assert sum_pos_axis(t) == Nx.sum(t, axes: [1])
      assert sum_neg_axis(t) == Nx.sum(t, axes: [-3])
      assert sum_pos_neg_axis(t) == Nx.sum(t, axes: [1, -3])
    end
  end

  describe "argmax/argmin" do
    defn argmax(t), do: Nx.argmax(t)
    defn argmin(t), do: Nx.argmin(t)
    defn argmax_axis(t), do: Nx.argmax(t, axis: 1)
    defn argmin_axis(t), do: Nx.argmin(t, axis: 1)
    defn argmax_high(t), do: Nx.argmax(t, axis: 1, tie_break: :high)
    defn argmin_high(t), do: Nx.argmin(t, axis: 1, tie_break: :high)

    test "computes the argmax across types" do
      assert argmax(Nx.tensor([1, 2, 3])) == Nx.tensor(2)
      assert argmax(Nx.tensor([1, 2, 3], type: {:s, 8})) == Nx.tensor(2, type: {:s, 8})
      assert argmax(Nx.tensor([1, 2, 3], type: {:u, 8})) == Nx.tensor(2, type: {:u, 8})
      assert argmax(Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor(2, type: {:f, 64})
      assert argmax(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})) == Nx.tensor(2, type: {:f, 32})
      assert argmax(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16})) == Nx.tensor(2, type: {:bf, 16})
      assert argmax(Nx.tensor([[1, 2, 3], [4, 5, 6]])) == Nx.tensor(5)
    end

    test "computes the argmin across types" do
      assert argmin(Nx.tensor([1, 2, 3])) == Nx.tensor(0)
      assert argmin(Nx.tensor([1, 2, 3], type: {:s, 8})) == Nx.tensor(0, type: {:s, 8})
      assert argmin(Nx.tensor([1, 2, 3], type: {:u, 8})) == Nx.tensor(0, type: {:u, 8})
      assert argmin(Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor(0, type: {:f, 64})
      assert argmin(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})) == Nx.tensor(0, type: {:f, 32})
      assert argmin(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16})) == Nx.tensor(0, type: {:bf, 16})
      assert argmin(Nx.tensor([[1, 2, 3], [4, 5, 6]])) == Nx.tensor(0)
    end

    test "computes the argmax on an axis" do
      assert argmax_axis(Nx.tensor([[[1, 1, 1], [1, 1, 3]], [[6, 2, 3], [2, 8, 3]]])) ==
               Nx.tensor([[0, 0, 1], [0, 1, 0]])
    end

    test "computes the argmin on an axis" do
      assert argmin_axis(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])) ==
               Nx.tensor([[1, 1, 0], [1, 0, 0]])
    end

    test "computes argmax with tie_break: :high" do
      assert argmax_axis(Nx.tensor([[1, 2, 2], [1, 2, 2]])) == Nx.tensor([1, 1])
      assert argmax_high(Nx.tensor([[1, 2, 2], [1, 2, 2]])) == Nx.tensor([2, 2])
    end
  end

  describe "dot product" do
    defn dot(a, b), do: Nx.dot(a, b)

    test "computes the dot product of scalars" do
      assert dot(Nx.tensor(2), Nx.tensor(2)) == Nx.tensor(4)
      assert dot(Nx.tensor(2.0), Nx.tensor(2.0)) == Nx.tensor(4.0)
      assert dot(Nx.tensor(-2.0), Nx.tensor(-2)) == Nx.tensor(4.0)
    end

    test "computes the dot product of vectors" do
      assert dot(Nx.tensor([1, 2, 3], type: {:s, 32}), Nx.tensor([4, 5, 6], type: {:s, 32})) ==
               Nx.tensor(32, type: {:s, 32})

      assert dot(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}), Nx.tensor([4, 5, 6])) ==
               Nx.tensor(32.0)

      assert dot(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([4.0, 5.0, 6.0])) == Nx.tensor(32.0)
    end

    test "computes the dot product of matrices" do
      assert dot(
               Nx.tensor([[1, 2, 3], [4, 5, 6]], type: {:s, 32}),
               Nx.tensor([[7, 8], [9, 10], [11, 12]], type: {:s, 32})
             ) ==
               Nx.tensor([[58, 64], [139, 154]], type: {:s, 32})

      assert dot(
               Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
               Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
             ) == Nx.tensor([[58.0, 64.0], [139.0, 154.0]])

      assert dot(
               Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
               Nx.tensor([[7, 8], [9, 10], [11, 12]])
             ) == Nx.tensor([[58.0, 64.0], [139.0, 154.0]])
    end

    test "computes the dot product of tensors" do
      assert dot(
               Nx.tensor(
                 [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                 type: {:s, 32}
               ),
               Nx.tensor(
                 [[[1, 2, 3], [3, 4, 5], [5, 6, 7]]],
                 type: {:s, 32}
               )
             ) ==
               Nx.tensor(
                 [
                   [[[22, 28, 34]], [[49, 64, 79]], [[76, 100, 124]]],
                   [[[22, 28, 34]], [[49, 64, 79]], [[76, 100, 124]]]
                 ],
                 type: {:s, 32}
               )
    end
  end

  describe "softmax" do
    defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

    test "computes softmax" do
      assert compare_tensors!(
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
      assert reshape_with_shape(Nx.tensor([1, 2, 3, 4])) == Nx.tensor([[1, 2], [3, 4]])
    end

    defn reshape_with_tensor(t, shape), do: Nx.reshape(t, shape)

    test "with tensor" do
      assert reshape_with_tensor(Nx.tensor([1, 2, 3, 4]), Nx.tensor([[0, 0], [0, 0]])) ==
               Nx.tensor([[1, 2], [3, 4]])

      assert reshape_with_tensor(Nx.tensor([1, 2, 3, 4]), Nx.tensor([[0], [0], [0], [0]])) ==
               Nx.tensor([[1], [2], [3], [4]])
    end

    defn reshape_with_constant(shape), do: Nx.reshape(123, shape)

    test "with constant" do
      assert reshape_with_constant(Nx.tensor([[[1]]])) == Nx.tensor([[[123]]])
    end
  end

  describe "assert_shape" do
    defn assert_shape_with_shape(t), do: Nx.assert_shape(t, {2, 2})

    test "with shape" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert assert_shape_with_shape(t) == t

      assert_raise ArgumentError,
                   "expected tensor with shape {2, 2} but tensor has shape {}",
                   fn ->
                     assert_shape_with_shape(Nx.tensor(1))
                   end
    end

    defn assert_shape_with_tensor(t, shape), do: Nx.assert_shape(t, shape, "oops")

    test "with tensor" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert assert_shape_with_tensor(t, t) == t

      assert_raise ArgumentError,
                   "expected tensor with shape {} but tensor has shape {2, 2} (oops)",
                   fn ->
                     assert_shape_with_tensor(t, Nx.tensor(1))
                   end
    end

    defn assert_shape_with_constant(shape), do: Nx.assert_shape(123, shape, "oops")

    test "with constant" do
      assert assert_shape_with_constant(Nx.tensor(1)) == Nx.tensor(123)

      assert_raise ArgumentError,
                   "expected tensor with shape {2} but tensor has shape {} (oops)",
                   fn ->
                     assert_shape_with_constant(Nx.tensor([1, 2]))
                   end
    end

    defn assert_shape_with_tuple(), do: Nx.assert_shape({1, 2, 3}, {}, "oops")

    test "with tuple" do
      assert_raise ArgumentError, "expected tensor with shape {} but got {1, 2, 3} (oops)", fn ->
        assert_shape_with_tuple()
      end
    end
  end

  describe "broadcast" do
    defn broadcast_with_shape(t), do: Nx.broadcast(t, {2, 2})

    test "with shape" do
      assert broadcast_with_shape(Nx.tensor([1, 2])) == Nx.tensor([[1, 2], [1, 2]])
      assert broadcast_with_shape(Nx.tensor([[1], [2]])) == Nx.tensor([[1, 1], [2, 2]])
    end

    defn broadcast_with_tensor(t, shape), do: Nx.broadcast(t, shape)

    test "with tensor" do
      tensors = [
        {Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([0, 0])},
        {Nx.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]]), Nx.tensor([0, 0])},
        {Nx.tensor([[1], [2]]), Nx.tensor([[0, 0]])},
        {Nx.tensor([[[1, 2]], [[3, 4]]]), Nx.tensor([[[0], [0]]])},
        {Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]),
         Nx.tensor([[[0], [0], [0]]])},
        {Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[[[0]]]])},
        {Nx.tensor([1, 2]), Nx.tensor([[[[0]]]])},
        {Nx.tensor([[1, 2]]), Nx.tensor([[[0], [0]], [[0], [0]]])},
        {Nx.tensor([[[1, 2]], [[3, 4]]]), Nx.tensor([[[[0], [0]], [[0], [0]]]])},
        {Nx.tensor([[[[1, 2]]], [[[3, 4]]]]), Nx.tensor([[[[0], [0]], [[0], [0]]]])},
        {Nx.tensor([[[1, 2]], [[3, 4]]]), Nx.tensor([[[0], [0]], [[0], [0]]])}
      ]

      for {left, right} <- tensors do
        assert add_two(left, right) == broadcast_with_tensor(left, right)
      end
    end

    defn broadcast_with_constant(shape), do: Nx.broadcast(123, shape)

    test "with constant" do
      assert broadcast_with_constant(Nx.tensor([[1, 2], [3, 4]])) ==
               Nx.tensor([[123, 123], [123, 123]])
    end
  end

  describe "random uniform" do
    defn random_uniform_fixed, do: Nx.random_uniform({30, 20})

    test "generates with shape" do
      t = random_uniform_fixed()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 64}

      for <<x::float-64-native <- Nx.Util.to_bitstring(t)>> do
        assert x >= 0.0 and x < 1
      end
    end

    defn random_uniform_min_max_int, do: Nx.random_uniform({30, 20}, 5, 10)
    defn random_uniform_min_max_float, do: Nx.random_uniform({30, 20}, 5.0, 10.0)

    test "generates with min/max" do
      t = random_uniform_min_max_int()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:s, 64}

      for <<x::signed-64-native <- Nx.Util.to_bitstring(t)>> do
        assert x >= 5 and x < 10
      end

      t = random_uniform_min_max_float()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 64}

      for <<x::float-64-native <- Nx.Util.to_bitstring(t)>> do
        assert x >= 5.0 and x < 10.0
      end
    end

    defn random_uniform_u32, do: Nx.random_uniform({30, 20}, 5, 10, type: {:u, 32})
    defn random_uniform_f32, do: Nx.random_uniform({30, 20}, 5.0, 10.0, type: {:f, 32})

    test "generates with type" do
      t = random_uniform_u32()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:u, 32}

      for <<x::unsigned-32-native <- Nx.Util.to_bitstring(t)>> do
        assert x >= 5 and x < 10
      end

      t = random_uniform_f32()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}

      for <<x::float-32-native <- Nx.Util.to_bitstring(t)>> do
        assert x >= 5.0 and x < 10.0
      end
    end

    defn random_uniform_tensor(t), do: Nx.random_uniform(t)
    defn random_uniform_tensor_with_type(t), do: Nx.random_uniform(t, type: {:f, 32})

    test "generates from tensor" do
      t = random_uniform_tensor(Nx.tensor([[1, 2], [3, 4]]))
      assert Nx.shape(t) == {2, 2}
      assert Nx.type(t) == {:f, 64}

      t = random_uniform_tensor_with_type(Nx.tensor([[1, 2, 3], [3, 4, 6]]))
      assert Nx.shape(t) == {2, 3}
      assert Nx.type(t) == {:f, 32}
    end
  end

  describe "random normal" do
    defn random_normal_fixed, do: Nx.random_normal({30, 20})

    test "generates with shape" do
      t = random_uniform_fixed()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 64}
    end

    defn random_normal_mu_sigma, do: Nx.random_normal({30, 20}, 5.0, 10.0)

    test "generates with mu/sigma" do
      t = random_normal_mu_sigma()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 64}
    end

    defn random_normal_f32, do: Nx.random_normal({30, 20}, 5.0, 10.0, type: {:f, 32})

    test "generates with type" do
      t = random_normal_f32()
      assert Nx.shape(t) == {30, 20}
      assert Nx.type(t) == {:f, 32}
    end

    defn random_normal_tensor(t), do: Nx.random_uniform(t)
    defn random_normal_tensor_with_type(t), do: Nx.random_uniform(t, type: {:f, 32})

    test "generates from tensor" do
      t = random_normal_tensor(Nx.tensor([[1, 2], [3, 4]]))
      assert Nx.shape(t) == {2, 2}
      assert Nx.type(t) == {:f, 64}

      t = random_normal_tensor_with_type(Nx.tensor([[1, 2, 3], [3, 4, 6]]))
      assert Nx.shape(t) == {2, 3}
      assert Nx.type(t) == {:f, 32}
    end
  end

  describe "iota" do
    defn iota_with_shape, do: Nx.iota({3, 4, 2, 3}, axis: 2)

    test "generates with shape" do
      assert iota_with_shape() == Nx.iota({3, 4, 2, 3}, axis: 2)
    end

    defn iota_with_type, do: Nx.iota({1, 2, 3}, axis: 1, type: {:f, 32})

    test "generates with type" do
      assert iota_with_type() == Nx.iota({1, 2, 3}, axis: 1, type: {:f, 32})
    end

    defn iota_from_tensor(t), do: Nx.iota(t, axis: 0)

    test "generates from tensor" do
      t = Nx.tensor([1, 2, 3])
      assert iota_from_tensor(t) == Nx.iota(t, axis: 0)
    end

    defn iota_no_axis, do: Nx.iota({2, 2, 2})

    test "generates without axis" do
      assert iota_no_axis() == Nx.iota({2, 2, 2})
    end

    defn iota_neg_axis, do: Nx.iota({2, 2, 2}, axis: -2)

    test "generates with negative axis" do
      assert iota_neg_axis() == Nx.iota({2, 2, 2}, axis: -2)
    end
  end

  describe "reflection" do
    defn random_from_type_and_shape(t),
      do: Nx.random_uniform(Nx.shape(t), 0, 10, type: Nx.type(t))

    test "type and shape" do
      t = random_from_type_and_shape(Nx.tensor([[1], [2]]))
      assert Nx.shape(t) == {2, 1}
      assert Nx.type(t) == {:s, 64}

      t = random_from_type_and_shape(Nx.tensor([[1], [2]], type: {:f, 32}))
      assert Nx.shape(t) == {2, 1}
      assert Nx.type(t) == {:f, 32}
    end

    defn rank_and_size(t), do: {Nx.rank(t), Nx.size(t)}

    test "rank and size" do
      assert rank_and_size(Nx.tensor([[1, 2], [3, 4]])) == {Nx.tensor(2), Nx.tensor(4)}
      assert rank_and_size(Nx.tensor([1, 2, 3, 4, 5])) == {Nx.tensor(1), Nx.tensor(5)}
    end

    defn rank_and_size_for_tuple(), do: {Nx.rank({8, 8}), Nx.size({8, 8})}

    test "rank and size for tuples" do
      assert rank_and_size_for_tuple() == {Nx.tensor(2), Nx.tensor(64)}
    end
  end

  describe "options" do
    @defn_compiler {Exla, keep_on_device: true}
    defn add_two_keep_on_device(a, b), do: a + b

    test "keeps data on device" do
      tensor = add_two_keep_on_device(1, 2)
      assert {Exla.NxDevice, {ref, :default}} = tensor.data
      assert is_reference(ref)
      assert tensor |> Nx.device_read() |> Nx.Util.to_bitstring() == <<3::64-native>>

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert {Exla.NxDevice, {ref, :default}} = tensor.data
      assert is_reference(ref)

      assert tensor |> Nx.device_read() |> Nx.Util.to_bitstring() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert tensor |> Nx.device_transfer() |> Nx.Util.to_bitstring() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   "Attempt to read from deallocated buffer.",
                   fn -> Nx.device_read(tensor) end
    end
  end

  # We need to round the floats because of imprecision between platforms
  defp compare_tensors!(
         %{type: {:f, size}, data: {dev, left_data}} = left,
         %{data: {dev, right_data}} = right
       ) do
    left_data = for <<x::float-size(size)-native <- left_data>>, do: Float.round(x, 5)
    right_data = for <<x::float-size(size)-native <- right_data>>, do: Float.round(x, 5)
    assert %{left | data: {dev, left_data}} == %{right | data: {dev, right_data}}
  end

  defp compare_tensors!(left, right) do
    assert left == right
  end
end
