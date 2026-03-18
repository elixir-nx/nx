defmodule Nx.EdgeCasesTest do
  @moduledoc """
  Regression tests for Nx edge cases:
  - window_scatter_max/min on f64
  - Nx.slice on scalar tensor
  - Nx.linspace with n=1
  - Nx.gather with scalar indices

  IEEE 754 overflow/domain/divzero tests are skipped pending
  upstream fix in the Complex library (elixir-nx/complex#29).
  """
  use ExUnit.Case, async: true

  # ── IEEE 754 tests (pending Complex library fix) ───────────────────
  # These tests require elixir-nx/complex#29 to be released.
  # Once Complex handles :math overflow/domain errors, these
  # will pass without any changes to BinaryBackend.

  describe "unary overflow returns Inf instead of crashing" do
    @tag :skip
    test "exp(large) returns Inf" do
      assert Nx.to_number(Nx.exp(Nx.tensor(1000.0))) == :infinity
    end

    @tag :skip
    test "expm1(large) returns Inf" do
      assert Nx.to_number(Nx.expm1(Nx.tensor(1000.0))) == :infinity
    end

    @tag :skip
    test "sinh(large positive) returns Inf" do
      assert Nx.to_number(Nx.sinh(Nx.tensor(1000.0))) == :infinity
    end

    @tag :skip
    test "sinh(large negative) returns -Inf" do
      assert Nx.to_number(Nx.sinh(Nx.tensor(-1000.0))) == :neg_infinity
    end

    @tag :skip
    test "cosh(large) returns Inf" do
      assert Nx.to_number(Nx.cosh(Nx.tensor(1000.0))) == :infinity
    end

    @tag :skip
    test "sigmoid(large positive) returns 1.0" do
      assert Nx.to_number(Nx.sigmoid(Nx.tensor(1.0e6))) == 1.0
    end

    @tag :skip
    test "sigmoid(large negative) returns 0.0" do
      assert Nx.to_number(Nx.sigmoid(Nx.tensor(-1.0e6))) == 0.0
    end
  end

  describe "domain errors return NaN instead of crashing" do
    @tag :skip
    test "asin outside [-1, 1]" do
      assert Nx.to_number(Nx.asin(Nx.tensor(2.0))) == :nan
    end

    @tag :skip
    test "acos outside [-1, 1]" do
      assert Nx.to_number(Nx.acos(Nx.tensor(2.0))) == :nan
    end

    @tag :skip
    test "acosh below 1" do
      assert Nx.to_number(Nx.acosh(Nx.tensor(0.5))) == :nan
    end

    @tag :skip
    test "atanh outside (-1, 1)" do
      assert Nx.to_number(Nx.atanh(Nx.tensor(2.0))) == :nan
    end

    @tag :skip
    test "atanh at boundaries returns Inf/-Inf" do
      assert Nx.to_number(Nx.atanh(Nx.tensor(1.0))) == :infinity
      assert Nx.to_number(Nx.atanh(Nx.tensor(-1.0))) == :neg_infinity
    end
  end

  describe "division by zero returns Inf/NaN instead of crashing" do
    @tag :skip
    test "positive / 0.0 = Inf" do
      assert Nx.to_number(Nx.divide(Nx.tensor(1.0), Nx.tensor(0.0))) == :infinity
    end

    @tag :skip
    test "negative / 0.0 = -Inf" do
      assert Nx.to_number(Nx.divide(Nx.tensor(-1.0), Nx.tensor(0.0))) == :neg_infinity
    end

    @tag :skip
    test "0.0 / 0.0 = NaN" do
      assert Nx.to_number(Nx.divide(Nx.tensor(0.0), Nx.tensor(0.0))) == :nan
    end

    @tag :skip
    test "normal division still works" do
      assert Nx.to_number(Nx.divide(Nx.tensor(10.0), Nx.tensor(2.0))) == 5.0
    end
  end

  # ── Active tests (fixes in this PR) ────────────────────────────────

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
