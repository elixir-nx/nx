defmodule Nx.TestingTest do
  use ExUnit.Case, async: true

  import Nx.Testing

  describe "assert_all_close/3" do
    test "passes on bit-identical tensors" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      assert_all_close(a, a)
    end

    test "passes on tensors within tolerance" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([1.00001, 2.00001, 3.00001])
      assert_all_close(a, b, atol: 1.0e-4)
    end

    test "passes on bit-identical vectorized tensors" do
      a = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      b = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      assert_all_close(a, b)
    end

    test "passes on vectorized tensors within tolerance" do
      a = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      b = Nx.tensor([1.00001, 2.00001]) |> Nx.vectorize(:foo)
      assert_all_close(a, b, atol: 1.0e-4)
    end

    test "error message includes max absolute difference" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([1.0, 2.5, 3.0])

      error =
        assert_raise ExUnit.AssertionError, fn ->
          assert_all_close(a, b, atol: 1.0e-4)
        end

      assert error.message =~ "max absolute difference"
      assert error.message =~ "0.5"
    end

    test "error message includes max-diff diagnostic for vectorized tensors" do
      # Exercises the case where both tensors have matching structure but
      # differ in values by more than the tolerance — the diagnostic should
      # include the max absolute difference so failures are still readable
      # even when the inspect output truncates to look identical.
      a = Nx.tensor([1.0, 2.0], type: :f32) |> Nx.vectorize(:foo)
      b = Nx.tensor([1.0, 2.0001], type: :f32) |> Nx.vectorize(:foo)

      error =
        assert_raise ExUnit.AssertionError, fn ->
          assert_all_close(a, b, atol: 0.0, rtol: 0.0)
        end

      assert error.message =~ "max absolute difference"
      assert error.message =~ "max relative difference"
    end

    test "error message shows a clear diagnostic for mismatched vec axes" do
      a = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      b = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:bar)

      error =
        assert_raise ExUnit.AssertionError, fn ->
          assert_all_close(a, b)
        end

      assert error.message =~ "vectorized_axes"
    end
  end

  describe "assert_equal/2" do
    test "passes on bit-identical tensors" do
      a = Nx.tensor([1, 2, 3])
      assert_equal(a, a)
    end

    test "passes on bit-identical vectorized tensors" do
      a = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      b = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      assert_equal(a, b)
    end

    test "error message includes max absolute difference" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([1.0, 2.5, 3.0])

      error =
        assert_raise ExUnit.AssertionError, fn ->
          assert_equal(a, b)
        end

      assert error.message =~ "max absolute difference"
      assert error.message =~ "0.5"
    end

    test "error message shows a clear diagnostic for mismatched vec axes" do
      a = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:foo)
      b = Nx.tensor([1.0, 2.0]) |> Nx.vectorize(:bar)

      error =
        assert_raise ExUnit.AssertionError, fn ->
          assert_equal(a, b)
        end

      assert error.message =~ "vectorized_axes"
    end

    test "handles NaN equality correctly" do
      a = Nx.tensor([:nan, 2.0, :nan])
      assert_equal(a, a)
    end
  end
end
