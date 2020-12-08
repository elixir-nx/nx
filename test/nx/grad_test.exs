defmodule Nx.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  describe "simple"  do
    defn grad_itself(t), do: grad(t, t)
    defn grad_constant(t), do: grad(t, 1.0)
    defn grad_unrelated(t, a), do: grad(t, a)

    test "computes gradient" do
      assert grad_itself(Nx.tensor(1.0)) == Nx.tensor(1.0)
      assert grad_constant(Nx.tensor(1.0)) == Nx.tensor(0.0)
      assert grad_unrelated(Nx.tensor(1.0), Nx.tensor(2.0)) == Nx.tensor(0.0)
    end
  end

  describe "addition rule" do
    defn addition_rule(t), do: grad(t, Nx.tanh(Nx.tanh(Nx.add(Nx.power(t, 2), Nx.power(t, 3)))))

    test "computes gradient" do
      assert addition_rule(Nx.tensor(1.0)) == Nx.tensor(0.1566267114813547)
    end
  end

  describe "product rule" do
    defn product_rule(t), do: grad(t, Nx.tanh(Nx.tanh(Nx.dot(Nx.power(t, 2), Nx.power(t, 3)))))

    test "computes gradient" do
      assert product_rule(Nx.tensor(1.0)) == Nx.tensor(1.2343397629215758)
    end
  end

  describe "power rule" do
    defn power_rule(t), do: grad(t, Nx.power(t, 3))

    test "computes gradient" do
      assert power_rule(Nx.tensor(5.0)) == Nx.tensor(75.0)
    end
  end

  describe "exponential rule" do
    defn exp_rule(t), do: grad(t, Nx.add(Nx.power(Nx.tanh(t), 2), Nx.power(Nx.tanh(t), 3)))

    test "computes gradient" do
      assert exp_rule(Nx.tensor(1.0)) == Nx.tensor(1.370487690448899)
    end
  end

  describe "tanh+exp" do
    defn grad_tanh(t), do: grad(t, Nx.tanh(t))
    defn grad_exp_tanh(t), do: grad(t, Nx.exp(Nx.tanh(t)))
    defn grad_tanh_exp(t), do: grad(t, Nx.tanh(Nx.exp(t)))
    defn grad_grad_tanh(t), do: grad(t, grad(t, Nx.tanh(t)))

    test "computes gradient" do
      assert grad_tanh(Nx.tensor(1.0)) == Nx.tensor(0.41997434161402614)
      assert grad_exp_tanh(Nx.tensor(1.0)) == Nx.tensor(0.8994538753454762)
      assert grad_tanh_exp(Nx.tensor(1.0)) == Nx.tensor(0.04693651986265914)
      assert grad_grad_tanh(Nx.tensor(1.0)) == Nx.tensor(-0.6397000084492246)
    end
  end
end