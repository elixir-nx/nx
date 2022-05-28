defmodule Nx.NonFiniteTest do
  @moduledoc """
  Responsible for testing all basic Nx functionality with
  complex number arguments.
  """
  use ExUnit.Case, async: true

  import Nx.Helpers
  import Nx, only: :sigils

  @arg Complex.new(:infinity, 3)
  @arg2 Complex.new(-2, 4)

  @inf_inf Complex.new(:infinity, :infinity)
  @nan_nan Complex.new(:nan, :nan)

  describe "unary operations" do
    test "exp" do
      assert Nx.exp(@arg) == Nx.tensor(@inf_inf)
    end

    test "expm1" do
      assert Nx.expm1(@arg) == Nx.tensor(@inf_inf)
    end

    test "log" do
      assert_all_close(Nx.log(@arg), Complex.new(:infinity, 0))
    end

    test "log1p" do
      assert_all_close(Nx.log1p(@arg), Complex.new(:infinity, 0))
    end

    test "sigmoid" do
      assert_all_close(Nx.sigmoid(@arg), Complex.new(1.0, 0.0))
    end

    test "cos" do
      assert_all_close(Nx.cos(@arg), @nan_nan)
    end

    test "sin" do
      assert_all_close(Nx.sin(@arg), @nan_nan)
    end

    test "tan" do
      assert_all_close(Nx.tan(@arg), @nan_nan)
    end

    test "cosh" do
      assert Nx.cosh(@arg) == Nx.tensor(@inf_inf)
    end

    test "sinh" do
      assert Nx.sinh(@arg) == Nx.tensor(@inf_inf)
    end

    test "tanh" do
      assert_all_close(Nx.tanh(@arg), @nan_nan)
    end

    test "acos" do
      assert_all_close(Nx.acos(@arg), @nan_nan)
    end

    test "asin" do
      assert_all_close(Nx.asin(@arg), @nan_nan)
    end

    test "atan" do
      assert_all_close(Nx.atan(@arg), @nan_nan)
    end

    test "acosh" do
      assert_all_close(Nx.acosh(@arg), Complex.new(:infinity, :math.pi() / 4))
    end

    test "asinh" do
      assert_all_close(Nx.asinh(@arg), Complex.new(:infinity, :math.pi() / 4))
    end

    test "atanh" do
      assert_all_close(Nx.atanh(@arg), @nan_nan)
    end

    test "sqrt" do
      assert_all_close(Nx.sqrt(@arg), Complex.new(:infinity, 0))
    end

    test "rsqrt" do
      assert_all_close(Nx.rsqrt(@arg), Complex.new(0.0, 0.0))
    end

    test "cbrt" do
      assert_all_close(Nx.cbrt(@arg), @nan_nan)
    end

    test "conjugate" do
      assert Nx.conjugate(@arg) == Nx.tensor(Complex.new(:infinity, -3))
    end

    test "phase" do
      assert_all_close(Nx.phase(@arg), Complex.phase(@arg))
    end

    for fun <- [:erf, :erfc, :erf_inv, :round, :floor, :ceil] do
      test "#{fun}" do
        assert_raise ArgumentError, "Nx.#{unquote(fun)}/1 does not support complex inputs", fn ->
          Nx.unquote(fun)(@arg)
        end
      end
    end

    test "random_uniform" do
      assert %Nx.Tensor{shape: {3, 3}, type: {:c, 64}} =
               t = Nx.random_uniform({3, 3}, type: {:c, 64})

      assert Enum.all?(Nx.to_flat_list(t), &is_struct(&1, Complex))
    end

    test "random_normal" do
      assert %Nx.Tensor{shape: {3, 3}, type: {:c, 64}} =
               t = Nx.random_normal({3, 3}, type: {:c, 64})

      assert Enum.all?(Nx.to_flat_list(t), &is_struct(&1, Complex))
    end
  end

  describe "aggregate operations" do
    test "all" do
      assert Nx.all(Nx.tensor([@arg, @arg2])) == Nx.tensor(1, type: {:u, 8})
    end

    test "all_close" do
      assert Nx.all_close(Nx.tensor([@arg, @arg2]), 0) == Nx.tensor(0, type: {:u, 8})
    end

    test "any" do
      assert Nx.any(Nx.tensor([@arg, @arg2])) == Nx.tensor(1, type: {:u, 8})
    end

    test "mean" do
      t = Nx.tensor([@arg, @arg2])

      mean =
        t
        |> Nx.sum()
        |> Nx.divide(2)

      assert Nx.mean(t) == mean
    end

    test "product" do
      assert Nx.product(Nx.tensor([@arg, @arg2])) == Nx.multiply(@arg, @arg2)
    end

    test "reduce" do
      assert Nx.reduce(Nx.tensor([@arg, @arg2]), 0, &Nx.add/2) == Nx.add(@arg, @arg2)
    end

    test "sum" do
      assert Nx.sum(Nx.tensor([@arg, @arg2])) == Nx.add(@arg, @arg2)
    end
  end

  describe "window operations" do
    test "window_mean" do
      t = Nx.tensor([@arg, @arg2])
      assert Nx.window_mean(t, {2})[0] == Nx.mean(t)
    end

    test "window_product" do
      t = Nx.tensor([@arg, @arg2])
      assert Nx.window_product(t, {2})[0] == Nx.product(t)
    end

    test "window_reduce" do
      t = Nx.tensor([@arg, @arg2])
      assert Nx.window_reduce(t, 0, {2}, &Nx.add/2)[0] == Nx.sum(t)
    end

    test "window_sum" do
      t = Nx.tensor([@arg, @arg2])
      assert Nx.window_sum(t, {2})[0] == Nx.sum(t)
    end
  end

  describe "binary operations" do
    test "add" do
      assert_all_close(Nx.add(@arg, @arg2), Complex.new(:infinity, 7))
    end

    test "subtract" do
      assert_all_close(Nx.subtract(@arg, @arg2), Complex.new(:infinity, -1))
    end

    test "multiply" do
      assert Nx.multiply(@arg, @arg2) == Nx.tensor(Complex.new(:neg_infinity, :infinity))
    end

    test "power" do
      assert_all_close(Nx.power(@arg, @arg2), @nan_nan)
    end

    test "divide" do
      assert Nx.divide(@arg, @arg2) == Nx.tensor(Complex.new(:neg_infinity, :neg_infinity))
    end

    test "atan2" do
      assert_all_close(Nx.atan2(Complex.new(7, 0), Complex.new(2.0, 0.0)), Complex.new(1.2925, 0))

      assert_raise ArithmeticError, "Complex.atan2 only accepts real numbers as arguments", fn ->
        Nx.atan2(7, @arg)
      end

      assert_raise ArithmeticError, "Complex.atan2 only accepts real numbers as arguments", fn ->
        Nx.atan2(@arg, 7)
      end
    end

    test "quotient" do
      assert_raise ArgumentError,
                   "quotient expects integer tensors as inputs and outputs an integer tensor, got: {:c, 64}",
                   fn ->
                     Nx.quotient(@arg, @arg2)
                   end
    end

    test "max" do
      # infinity as right arg
      assert ~V[Inf] == Nx.max(~V[-Inf], ~V[Inf])
      assert ~V[Inf] == Nx.max(~V[1], ~V[Inf])
      assert ~V[NaN] == Nx.max(~V[NaN], ~V[Inf])
      assert ~V[Inf] == Nx.max(~V[Inf], ~V[Inf])

      # neg_inf as right arg
      assert ~V[-Inf] == Nx.max(~V[-Inf], ~V[-Inf])
      assert ~V[1.0] == Nx.max(~V[1], ~V[-Inf])
      assert ~V[NaN] == Nx.max(~V[NaN], ~V[-Inf])
      assert ~V[Inf] == Nx.max(~V[Inf], ~V[-Inf])

      # nan as right arg
      assert ~V[NaN] == Nx.max(~V[-Inf], ~V[NaN])
      assert ~V[NaN] == Nx.max(~V[1], ~V[NaN])
      assert ~V[NaN] == Nx.max(~V[NaN], ~V[NaN])
      assert ~V[NaN] == Nx.max(~V[Inf], ~V[NaN])
    end

    test "min" do
      # infinity as right arg
      assert ~V[-Inf] == Nx.min(~V[-Inf], ~V[Inf])
      assert ~V[1.0] == Nx.min(~V[1], ~V[Inf])
      assert ~V[NaN] == Nx.min(~V[NaN], ~V[Inf])
      assert ~V[Inf] == Nx.min(~V[Inf], ~V[Inf])

      # neg_inf as right arg
      assert ~V[-Inf] == Nx.min(~V[-Inf], ~V[-Inf])
      assert ~V[-Inf] == Nx.min(~V[1], ~V[-Inf])
      assert ~V[NaN] == Nx.min(~V[NaN], ~V[-Inf])
      assert ~V[-Inf] == Nx.min(~V[Inf], ~V[-Inf])

      # nan as right arg
      assert ~V[NaN] == Nx.min(~V[-Inf], ~V[NaN])
      assert ~V[NaN] == Nx.min(~V[1], ~V[NaN])
      assert ~V[NaN] == Nx.min(~V[NaN], ~V[NaN])
      assert ~V[NaN] == Nx.min(~V[Inf], ~V[NaN])
    end

    test "remainder" do
      assert_raise ArgumentError,
                   "errors were found at the given arguments:\n\n  * 1st argument: not a number\n  * 2nd argument: not a number\n",
                   fn ->
                     Nx.remainder(
                       Nx.tensor(:neg_infinity, type: {:f, 32}),
                       Nx.tensor(:infinity, type: {:f, 32})
                     )
                   end
    end

    for op <- [
          :bitwise_and,
          :bitwise_or,
          :bitwise_xor,
          :left_shift,
          :right_shift
        ] do
      test "#{op}" do
        assert_raise ArgumentError,
                     "bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}",
                     fn ->
                       Nx.unquote(op)(
                         Nx.tensor(:infinity, type: {:f, 32}),
                         Nx.tensor(:infinity, type: {:f, 32})
                       )
                     end
      end
    end
  end
end
