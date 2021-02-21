defmodule Nx.SharedTest do
  use ExUnit.Case, async: true
  doctest Nx.Shared

  if Nx.Shared.math_fun_supported?(:erf, 1) do
    describe "erf/1" do
      test "matches :math implementation" do
        for _ <- 0..20 do
          x = :rand.uniform() * 100
          a = :math.erf(x)
          b = Nx.Shared.erf(x)
          assert_in_delta(a, b, 0.0000001)
        end
      end
    end
  end
end
