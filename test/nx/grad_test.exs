defmodule Nx.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.GradHelpers
  doctest Nx.Defn.Grad

  describe "simple" do
    defn grad_itself(t), do: grad(t, t)
    defn grad_constant(t), do: grad(t, 1.0)
    defn grad_unrelated(t, a), do: grad(t, a)

    test "computes gradient for scalars" do
      assert grad_itself(Nx.tensor(1.0)) == Nx.tensor(1.0)
      assert grad_constant(Nx.tensor(1.0)) == Nx.tensor(0.0)
      assert grad_unrelated(Nx.tensor(1.0), Nx.tensor(2.0)) == Nx.tensor(0.0)
    end

    test "computes gradient for tensors" do
      assert grad_constant(Nx.tensor([1.0, 2.0, 3.0])) ==
               Nx.tensor([0.0, 0.0, 0.0])

      assert grad_unrelated(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor(2.0)) ==
               Nx.tensor([0.0, 0.0, 0.0])
    end
  end

  describe "chain rule" do
    defn grad_tanh_exp(t), do: grad(t, Nx.tanh(Nx.exp(t)))

    test "computes gradient" do
      assert grad_tanh_exp(Nx.tensor(1.0)) == Nx.tensor(0.04693651986265914)

      for _ <- 1..100 do
        t = Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        check_grads!(&Nx.tanh(Nx.exp(&1)), &grad_tanh_exp/1, t)
      end
    end
  end
end
