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

  describe "addition rule" do
    defn addition_rule(t), do: Nx.tanh(Nx.tanh(Nx.add(Nx.power(t, 2), Nx.power(t, 3))))
    defn grad_addition_rule(t), do: grad(t, addition_rule(t))

    test "computes gradient of complex rules" do
      assert grad_addition_rule(Nx.tensor(1.0)) == Nx.tensor(0.1566267114813547)

      for _ <- 1..100 do
        check_grads!(
          &addition_rule/1,
          &grad_addition_rule/1,
          Nx.random_uniform({}, 0.0, 1000.0, type: {:f, 64})
        )
      end
    end
  end

  describe "product rule" do
    defn product_rule(t), do: Nx.tanh(Nx.tanh(Nx.multiply(Nx.power(t, 2), Nx.power(t, 3))))
    defn grad_product_rule(t), do: grad(t, product_rule(t))

    test "computes gradient for scalars" do
      assert grad_product_rule(Nx.tensor(1.0)) == Nx.tensor(1.2343397629215758)

      for _ <- 1..100 do
        check_grads!(
          &product_rule/1,
          &grad_product_rule/1,
          Nx.random_uniform({}, 0.0, 1000.0, type: {:f, 64})
        )
      end
    end

    defn sum_product_rule(t), do: Nx.sum(Nx.multiply(Nx.power(t, 2), Nx.power(t, 3)))
    defn grad_sum_product_rule(t), do: grad(t, sum_product_rule(t))

    test "computes gradient for tensors" do
      assert grad_sum_product_rule(Nx.tensor([[1, 2], [3, 4]])) ==
               Nx.tensor([[5.0, 80.0], [405.0, 1280.0]])
    end
  end

  describe "division rule" do
    defn division_rule(t), do: Nx.divide(Nx.tanh(t), t)
    defn grad_division_rule(t), do: grad(t, division_rule(t))

    test "computes gradient" do
      assert grad_division_rule(Nx.tensor(1.0)) == Nx.tensor(-0.3416198143417387)

      for _ <- 1..100 do
        check_grads!(
          &division_rule/1,
          &grad_division_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn division_num_rule(t), do: Nx.divide(Nx.tanh(t), 2)
    defn grad_division_num_rule(t), do: grad(t, division_num_rule(t))

    test "computes gradient for constant denominator" do
      for _ <- 1..100 do
        check_grads!(
          &division_num_rule/1,
          &grad_division_num_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn division_den_rule(t), do: Nx.divide(2, Nx.exp(t))
    defn grad_division_den_rule(t), do: grad(t, division_den_rule(t))

    test "computes gradient for constant numerator" do
      for _ <- 1..100 do
        check_grads!(
          &division_den_rule/1,
          &grad_division_den_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "remainder rule" do
    defn remainder_rule(t), do: Nx.remainder(Nx.tanh(t), t)
    defn grad_remainder_rule(t), do: grad(t, remainder_rule(t))

    test "computes gradient" do
      assert grad_remainder_rule(Nx.tensor(1.0)) == Nx.tensor(0.41997434161402614)

      for _ <- 1..100 do
        check_grads!(
          &remainder_rule/1,
          &grad_remainder_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn remainder_num_rule(t), do: Nx.remainder(Nx.tanh(t), 2)
    defn grad_remainder_num_rule(t), do: grad(t, remainder_num_rule(t))

    test "computes gradient for constant denominator" do
      for _ <- 1..100 do
        check_grads!(
          &remainder_num_rule/1,
          &grad_remainder_num_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn remainder_den_rule(t), do: Nx.remainder(2, Nx.exp(t))
    defn grad_remainder_den_rule(t), do: grad(t, remainder_den_rule(t))

    test "computes gradient for constant numerator" do
      for _ <- 1..100 do
        check_grads!(
          &remainder_den_rule/1,
          &grad_remainder_den_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "power rule" do
    defn power_rule(t), do: Nx.power(t, 3)
    defn grad_power_rule(t), do: grad(t, power_rule(t))

    test "computes gradient" do
      assert grad_power_rule(Nx.tensor(5.0)) == Nx.tensor(75.0)

      for _ <- 1..100 do
        check_grads!(
          &power_rule/1,
          &grad_power_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "exponential rule" do
    defn exp_rule(t), do: Nx.add(Nx.power(Nx.tanh(t), 2), Nx.power(Nx.tanh(t), 3))
    defn grad_exp_rule(t), do: grad(t, exp_rule(t))

    test "computes gradient" do
      assert grad_exp_rule(Nx.tensor(1.0)) == Nx.tensor(1.370487690448899)

      for _ <- 1..100 do
        check_grads!(
          &exp_rule/1,
          &grad_exp_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "arctan2 rule" do
    defn arctan2_rule(t), do: Nx.arctan2(Nx.tanh(t), t)
    defn grad_arctan2_rule(t), do: grad(t, arctan2_rule(t))

    test "computes gradient" do
      assert grad_arctan2_rule(Nx.tensor(1.0)) == Nx.tensor(-0.2162115612038287)

      for _ <- 1..100 do
        check_grads!(
          &arctan2_rule/1,
          &grad_arctan2_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
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

  describe "grad grad" do
    defn grad_tanh_base(t), do: grad(t, Nx.tanh(t))
    defn grad_grad_tanh(t), do: grad(t, grad(t, Nx.tanh(t)))

    test "computes gradient" do
      assert grad_grad_tanh(Nx.tensor(1.0)) == Nx.tensor(-0.6397000084492246)

      for _ <- 1..100 do
        t = Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        check_grads!(&grad_tanh_base/1, &grad_grad_tanh/1, t)
      end
    end
  end
end
