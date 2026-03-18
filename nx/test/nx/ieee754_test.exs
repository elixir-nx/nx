defmodule Nx.IEEE754Test do
  @moduledoc """
  Regression tests for IEEE 754 compliance in BinaryBackend.

  These tests verify that BinaryBackend returns Inf/NaN instead of
  crashing with ArithmeticError for overflow, domain errors, and
  division by zero. Also covers linspace n=1, scalar slice, gather
  scalar indices, and window_scatter f64.
  """
  use ExUnit.Case, async: true

  describe "unary overflow returns Inf instead of crashing" do
    test "exp(large) returns Inf" do
      assert Nx.to_number(Nx.exp(Nx.tensor(1000.0))) == :infinity
    end

    test "expm1(large) returns Inf" do
      assert Nx.to_number(Nx.expm1(Nx.tensor(1000.0))) == :infinity
    end

    test "sinh(large positive) returns Inf" do
      assert Nx.to_number(Nx.sinh(Nx.tensor(1000.0))) == :infinity
    end

    test "sinh(large negative) returns -Inf" do
      assert Nx.to_number(Nx.sinh(Nx.tensor(-1000.0))) == :neg_infinity
    end

    test "cosh(large) returns Inf" do
      assert Nx.to_number(Nx.cosh(Nx.tensor(1000.0))) == :infinity
    end

    test "sigmoid(large positive) returns 1.0" do
      assert Nx.to_number(Nx.sigmoid(Nx.tensor(1.0e6))) == 1.0
    end

    test "sigmoid(large negative) returns 0.0" do
      assert Nx.to_number(Nx.sigmoid(Nx.tensor(-1.0e6))) == 0.0
    end
  end

  describe "domain errors return NaN instead of crashing" do
    test "asin outside [-1, 1]" do
      assert Nx.to_number(Nx.asin(Nx.tensor(2.0))) == :nan
      assert Nx.to_number(Nx.asin(Nx.tensor(-2.0))) == :nan
    end

    test "acos outside [-1, 1]" do
      assert Nx.to_number(Nx.acos(Nx.tensor(2.0))) == :nan
      assert Nx.to_number(Nx.acos(Nx.tensor(-2.0))) == :nan
    end

    test "acosh below 1" do
      assert Nx.to_number(Nx.acosh(Nx.tensor(0.5))) == :nan
    end

    test "atanh outside (-1, 1)" do
      assert Nx.to_number(Nx.atanh(Nx.tensor(2.0))) == :nan
      assert Nx.to_number(Nx.atanh(Nx.tensor(-2.0))) == :nan
    end

    test "atanh at boundaries returns Inf/-Inf" do
      assert Nx.to_number(Nx.atanh(Nx.tensor(1.0))) == :infinity
      assert Nx.to_number(Nx.atanh(Nx.tensor(-1.0))) == :neg_infinity
    end
  end

  describe "normal values still work after overflow fix" do
    test "exp(0) == 1" do
      assert Nx.to_number(Nx.exp(Nx.tensor(0.0))) == 1.0
    end

    test "sin(1) is correct" do
      assert_in_delta Nx.to_number(Nx.sin(Nx.tensor(1.0))), :math.sin(1.0), 1.0e-6
    end

    test "asin(0.5) is correct" do
      assert_in_delta Nx.to_number(Nx.asin(Nx.tensor(0.5))), :math.asin(0.5), 1.0e-6
    end

    test "sigmoid(0) == 0.5" do
      assert_in_delta Nx.to_number(Nx.sigmoid(Nx.tensor(0.0))), 0.5, 1.0e-6
    end
  end

  describe "division by zero returns Inf/NaN instead of crashing" do
    test "positive / 0.0 = Inf" do
      assert Nx.to_number(Nx.divide(Nx.tensor(1.0), Nx.tensor(0.0))) == :infinity
    end

    test "negative / 0.0 = -Inf" do
      assert Nx.to_number(Nx.divide(Nx.tensor(-1.0), Nx.tensor(0.0))) == :neg_infinity
    end

    test "0.0 / 0.0 = NaN" do
      assert Nx.to_number(Nx.divide(Nx.tensor(0.0), Nx.tensor(0.0))) == :nan
    end

    test "positive / -0.0 = -Inf" do
      assert Nx.to_number(Nx.divide(Nx.tensor(1.0), Nx.tensor(-0.0))) == :neg_infinity
    end

    test "negative / -0.0 = Inf" do
      assert Nx.to_number(Nx.divide(Nx.tensor(-1.0), Nx.tensor(-0.0))) == :infinity
    end

    test "normal division still works" do
      assert Nx.to_number(Nx.divide(Nx.tensor(10.0), Nx.tensor(2.0))) == 5.0
    end
  end

  describe "window_scatter_max/min on f64" do
    test "window_scatter_max works with f64" do
      t = Nx.iota({6}, type: :f64)
      s = Nx.iota({3}, type: :f64)
      init = Nx.tensor(0.0, type: :f64)
      result = Nx.window_scatter_max(t, s, init, {2}, strides: [2], padding: :valid)
      assert Nx.type(result) == {:f, 64}
      assert Nx.shape(result) == {6}
    end

    test "window_scatter_min works with f64" do
      t = Nx.iota({6}, type: :f64)
      s = Nx.iota({3}, type: :f64)
      init = Nx.tensor(0.0, type: :f64)
      result = Nx.window_scatter_min(t, s, init, {2}, strides: [2], padding: :valid)
      assert Nx.type(result) == {:f, 64}
      assert Nx.shape(result) == {6}
    end
  end

  describe "scalar slice" do
    test "slice of scalar tensor returns scalar" do
      t = Nx.tensor(42)
      result = Nx.slice(t, [], [])
      assert Nx.to_number(result) == 42
    end

    test "scalar slice with f64" do
      t = Nx.tensor(3.14, type: :f64)
      result = Nx.slice(t, [], [])
      assert_in_delta Nx.to_number(result), 3.14, 1.0e-10
    end
  end

  describe "linspace n=1" do
    test "linspace n=1 returns start value" do
      result = Nx.linspace(0, 10, n: 1)
      assert Nx.shape(result) == {1}
      assert Nx.to_flat_list(result) == [0.0]
    end

    test "linspace n=1 with same start/stop" do
      result = Nx.linspace(5, 5, n: 1)
      assert Nx.to_flat_list(result) == [5.0]
    end

    test "linspace n=2 still works" do
      result = Nx.linspace(0, 10, n: 2)
      assert Nx.to_flat_list(result) == [0.0, 10.0]
    end
  end

  describe "gather scalar indices error" do
    test "gather raises correct error on scalar indices" do
      assert_raise ArgumentError, ~r/expected indices rank to be at least 1/, fn ->
        Nx.gather(Nx.iota({3}), Nx.tensor(0))
      end
    end

    test "gather with valid indices still works" do
      t = Nx.iota({3, 4})
      result = Nx.gather(t, Nx.tensor([[0, 0], [2, 3]]))
      assert Nx.to_flat_list(result) == [0, 11]
    end
  end
end
