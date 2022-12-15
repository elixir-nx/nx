defmodule Nx.ComplexTest do
  @moduledoc """
  Responsible for testing all basic Nx functionality with
  complex number arguments.
  """
  use ExUnit.Case, async: true

  import Nx.Helpers

  @arg Complex.new(2, 3)
  @arg2 Complex.new(-2, 7)

  describe "unary operations" do
    test "exp" do
      assert_all_close(Nx.exp(@arg), Complex.new(-7.315, 1.042))
    end

    test "expm1" do
      assert_all_close(Nx.expm1(@arg), Complex.new(-8.315, 1.042))
    end

    test "log" do
      assert_all_close(Nx.log(@arg), Complex.new(1.2824, 0.98279))
    end

    test "log1p" do
      assert_all_close(Nx.log1p(@arg), Complex.new(1.4451, 0.7854))
    end

    test "sigmoid" do
      assert_all_close(Nx.sigmoid(@arg), Complex.new(1.1541, 0.0254))
    end

    test "cos" do
      assert_all_close(Nx.cos(@arg), Complex.new(-4.1896, -9.10925))
    end

    test "sin" do
      assert_all_close(Nx.sin(@arg), Complex.new(9.1544, -4.1689))
    end

    test "tan" do
      assert_all_close(Nx.tan(@arg), Complex.new(-0.00376, 1.00323))
    end

    test "cosh" do
      assert_all_close(Nx.cosh(@arg), Complex.new(-3.72454, 0.51182))
    end

    test "sinh" do
      assert_all_close(Nx.sinh(@arg), Complex.new(-3.59056, 0.53092))
    end

    test "tanh" do
      assert_all_close(Nx.tanh(@arg), Complex.new(0.96538, -0.00988))
    end

    test "acos" do
      assert_all_close(Nx.acos(@arg), Complex.new(1.0001, -1.9833))
    end

    test "asin" do
      assert_all_close(Nx.asin(@arg), Complex.new(0.57065, 1.98338))
    end

    test "atan" do
      assert_all_close(Nx.atan(@arg), Complex.new(1.40992, 0.22907))
    end

    test "acosh" do
      assert_all_close(Nx.acosh(@arg), Complex.new(1.9833, 1.0001))
    end

    test "asinh" do
      assert_all_close(Nx.asinh(@arg), Complex.new(1.9686, 0.96465))
    end

    test "atanh" do
      assert_all_close(Nx.atanh(@arg), Complex.new(0.14694, 1.3389))
    end

    test "sqrt" do
      assert_all_close(Nx.sqrt(@arg), Complex.new(1.6741, 0.8959))
    end

    test "rsqrt" do
      assert_all_close(Nx.rsqrt(@arg), Complex.new(0.4643, -0.2485))
    end

    test "cbrt" do
      assert_all_close(Nx.cbrt(@arg), Complex.new(1.4518, 0.4934))
    end

    test "conjugate" do
      assert Nx.conjugate(@arg) == Nx.tensor(Complex.new(2, -3))
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
      assert_all_close(Nx.add(@arg, @arg2), Complex.new(0, 10))
    end

    test "subtract" do
      assert_all_close(Nx.subtract(@arg, @arg2), Complex.new(4, -4))
    end

    test "multiply" do
      assert_all_close(Nx.multiply(@arg, @arg2), Complex.new(-25, 8))
    end

    test "power" do
      assert_all_close(Nx.power(@arg, @arg2), Complex.new(5.90369e-5, 5.26792e-5))
    end

    test "divide" do
      assert_all_close(Nx.divide(@arg, @arg2), Complex.new(0.32075, -0.37735))
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

    for op <- [:remainder, :max, :min] do
      test "#{op}" do
        assert_raise ArgumentError, "Nx.#{unquote(op)}/2 does not support complex inputs", fn ->
          Nx.unquote(op)(@arg, @arg2)
        end
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
                     "bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:c, 64}",
                     fn ->
                       Nx.unquote(op)(@arg, @arg2)
                     end
      end
    end
  end

  describe "LinAlg not yet implemented" do
    for function <- [:svd] do
      test "#{function}" do
        t = Nx.broadcast(Nx.tensor(1, type: {:c, 64}), {3, 3})

        assert_raise ArgumentError,
                     "Nx.LinAlg.#{unquote(function)}/2 is not yet implemented for complex inputs",
                     fn ->
                       Nx.LinAlg.unquote(function)(t)
                     end
      end
    end

    test "triangular_solve fails with singular complex matrix" do
      t = Nx.broadcast(Nx.tensor(0, type: {:c, 64}), {3, 3})

      assert_raise ArgumentError, "can't solve for singular matrix", fn ->
        Nx.LinAlg.triangular_solve(t, t)
      end
    end
  end
end
