defmodule Nx.SharedTest do
  use ExUnit.Case, async: true

  if Nx.Shared.math_func_supported?(:atanh, 1) do
    describe "atanh_fallback/1" do
      test "matches :math implemenation" do
        for _ <- 0..20 do
          # domain: (-1, 1)
          x = :rand.uniform() * 1.9999 - 1.0
          a = :math.atanh(x)
          b = Nx.Shared.atanh_fallback(x)
          assert_in_delta(a, b, 0.0000001)
        end
      end
    end
  end

  if Nx.Shared.math_func_supported?(:acosh, 1) do
    describe "acosh_fallback/1" do
      test "matches :math implemenation" do
        for _ <- 0..20 do
          # domain: [1, :infinity)
          x = :rand.uniform() + 1.0 * 9_999_999
          a = :math.acosh(x)
          b = Nx.Shared.acosh_fallback(x)
          assert_in_delta(a, b, 0.0000001)
        end
      end
    end
  end

  if Nx.Shared.math_func_supported?(:asinh, 1) do
    describe "asinh_fallback/1" do
      test "matches :math implemenation" do
        for _ <- 0..20 do
          x = :rand.uniform() * 100
          a = :math.asinh(x)
          b = Nx.Shared.asinh_fallback(x)
          assert_in_delta(a, b, 0.0000001)
        end
      end
    end
  end

  if Nx.Shared.math_func_supported?(:erf, 1) do
    describe "erf_fallback/1" do
      test "matches :math implemenation" do
        for _ <- 0..20 do
          x = :rand.uniform() * 100
          a = :math.erf(x)
          b = Nx.Shared.erf_fallback(x)
          assert_in_delta(a, b, 0.0000001)
        end
      end
    end
  end
end
