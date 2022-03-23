defmodule Nx.ComplexTest do
  @moduledoc """
  Responsible for testing all basic Nx functionality with
  complex number arguments.
  """
  use ExUnit.Case, async: true

  import Nx.Helpers

  @arg Complex.new(2, 3)

  describe "unary options" do
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

    test "logistic" do
      assert_all_close(Nx.logistic(@arg), Complex.new(1.1541, 0.0254))
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

    for fun <- [:erf, :erfc, :erf_inv] do
      test "#{fun}" do
        assert_raise ArgumentError, "Nx.#{unquote(fun)}/1 does not support complex numbers", fn ->
          Nx.unquote(fun)(@arg)
        end
      end
    end
  end
end
