defmodule Torchx.ComplexTest do
  @moduledoc """
  Responsible for testing all basic Nx functionality with
  complex number arguments.
  """
  use Torchx.Case, async: true

  import Nx, only: :sigils

  @arg Complex.new(2, 3)
  @arg2 Complex.new(-2, 7)

  describe "creation" do
    @tag :focus
    test "constant" do
      t = Nx.tensor(@arg)
      assert {:c, 64} == t.type
      assert @arg == Nx.to_number(t)
    end

    test "complex-only list" do
      l = [@arg, @arg2]
      t = Nx.tensor(l)
      assert {:c, 64} == t.type
      assert l == Nx.to_flat_list(t)
    end

    test "mixed list" do
      l = [1, @arg, @arg2]
      t = Nx.tensor(l)
      assert {:c, 64} == t.type
      assert [Complex.new(1), @arg, @arg2] == Nx.to_flat_list(t)
    end
  end

  describe "unary operations" do
    test "exp" do
      assert_all_close(Nx.exp(@arg), Complex.new(-7.315, 1.042))
    end

    test "expm1" do
      assert_raise ArithmeticError, "Torchx does not support complex values for expm1", fn ->
        Nx.expm1(@arg)
      end
    end

    test "log" do
      assert_all_close(Nx.log(@arg), Complex.new(1.2824, 0.98279))
    end

    test "log1p" do
      assert_raise ArithmeticError, "Torchx does not support complex values for log1p", fn ->
        Nx.log1p(@arg)
      end
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
      assert_all_close(Nx.pow(@arg, @arg2), Complex.new(5.90369e-5, 5.26792e-5))
    end

    test "divide" do
      assert_all_close(Nx.divide(@arg, @arg2), Complex.new(0.32075, -0.37735))
    end

    test "atan2" do
      assert_raise ArithmeticError, "Torchx does not support complex values for atan2", fn ->
        Nx.atan2(7, @arg)
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

    test "invert" do
      a = ~M[
        1 0 i
        0 -1i 0
        0 0 2
      ]

      expected_result = ~M[
        1 0 -0.5i
        0 1i 0
        0 0 0.5
      ]

      result = Nx.LinAlg.invert(a)

      assert_all_close(result, expected_result)

      assert_all_close(Nx.dot(a, result), Nx.eye(Nx.shape(a)))
      assert_all_close(Nx.dot(result, a), Nx.eye(Nx.shape(a)))
    end

    test "solve" do
      a = ~M[
        1 0 i
       -1i 0 1i
        1 1 1
      ]

      b = ~V[3+i 4 2-2i]

      result = ~V[i 2 -3i]

      assert_all_close(Nx.LinAlg.solve(a, b), result)
    end
  end

  describe "matrix_power" do
    test "supports complex with positive exponent" do
      a = ~M[
        1 1i
        -1i 1
      ]

      n = 5

      assert_all_close(Nx.LinAlg.matrix_power(a, n), Nx.multiply(2 ** (n - 1), a))
    end

    test "supports complex with 0 exponent" do
      a = ~M[
        1 1i
        -1i 1
      ]

      assert_all_close(Nx.LinAlg.matrix_power(a, 0), Nx.eye(Nx.shape(a)))
    end

    test "supports complex with negative exponent" do
      a = ~M[
        1 -0.5i
        0 0.5
      ]

      result = ~M[
        1 15i
        0 16
      ]

      assert_all_close(Nx.LinAlg.matrix_power(a, -4), result)
    end
  end
end
