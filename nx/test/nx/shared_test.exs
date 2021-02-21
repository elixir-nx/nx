defmodule Nx.SharedTest do
  use ExUnit.Case, async: true
  doctest Nx.Shared

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
