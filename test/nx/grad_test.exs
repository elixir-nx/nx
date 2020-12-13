defmodule Nx.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.GradHelpers

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

    test "computes gradient" do
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
    defn product_rule(t), do: Nx.tanh(Nx.tanh(Nx.dot(Nx.power(t, 2), Nx.power(t, 3))))
    defn grad_product_rule(t), do: grad(t, product_rule(t))

    test "computes gradient" do
      assert grad_product_rule(Nx.tensor(1.0)) == Nx.tensor(1.2343397629215758)

      for _ <- 1..100 do
        check_grads!(
          &product_rule/1,
          &grad_product_rule/1,
          Nx.random_uniform({}, 0.0, 1000.0, type: {:f, 64})
        )
      end
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

  describe "tuples" do
    defnp tuple_pattern({a, b}), do: Nx.power(a, 2) + b
    defn grad_tuple_pattern(t), do: grad(t, tuple_pattern({t, 2.0}))

    test "as patterns" do
      assert grad_tuple_pattern(Nx.tensor(1.0)) == Nx.tensor(2.0)
    end

    defn grad_tuple_input(a, b) do
      grad({a, b}, Nx.power(a, 2) * Nx.power(b, 3))
    end

    defn grad_tuple_input(a, b, c) do
      grad({a, b, c}, Nx.power(a, 2) * Nx.power(b, 3) * Nx.power(c, 4))
    end

    test "as multiple inputs" do
      assert grad_tuple_input(Nx.tensor(1.0), Nx.tensor(1.0)) ==
               {Nx.tensor(2.0), Nx.tensor(3.0)}

      assert grad_tuple_input(Nx.tensor(1.0), Nx.tensor(1.0), Nx.tensor(1.0)) ==
               {Nx.tensor(2.0), Nx.tensor(3.0), Nx.tensor(4.0)}
    end
  end

  describe "tensor constant" do
    @one_two_three Nx.tensor(123)
    defn grad_tensor_constant(t), do: grad(t, @one_two_three)
    defn grad_tensor_power_plus_constant(t), do: grad(t, Nx.power(t, 2) + @one_two_three)

    test "computes gradient for scalars" do
      assert grad_tensor_constant(Nx.tensor(1.0)) == Nx.tensor(0.0)
      assert grad_tensor_power_plus_constant(Nx.tensor(1.0)) == Nx.tensor(2.0)
    end

    test "computes gradient for tensors" do
      assert grad_tensor_constant(Nx.tensor([1.0, 2.0, 3.0])) == Nx.tensor([0.0, 0.0, 0.0])
    end
  end

  describe "assert_shape" do
    defn grad_assert(t), do: grad(t, t)

    test "raises on invalid return" do
      assert_raise ArgumentError,
                   ~r"expected tensor with shape \{\} but tensor has shape \{2\}",
                   fn -> grad_assert(Nx.tensor([1, 2])) end
    end
  end

  describe "broadcast" do
    defn grad_sum_broadcast(t), do: grad(t, Nx.sum(Nx.broadcast(t, {2, 2})))

    test "computes gradient" do
      assert grad_sum_broadcast(Nx.tensor([[0.0, 1.0], [2.0, 3.0]])) ==
               Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      assert grad_sum_broadcast(Nx.tensor([0.0, 1.0])) ==
               Nx.tensor([2.0, 2.0])

      assert grad_sum_broadcast(Nx.tensor(0.0)) ==
               Nx.tensor(4.0)
    end
  end

  for fun <-
        [:cbrt, :cos, :exp, :expm1, :log, :log1p, :logistic] ++
          [:negate, :rsqrt, :sin, :sqrt, :sum, :tanh] do
    describe "#{fun}" do
      grad_fun = :"grad_#{fun}"
      defn unquote(grad_fun)(t), do: grad(t, Nx.unquote(fun)(t))

      test "computes gradient" do
        for _ <- 1..100 do
          t = Nx.random_uniform({}, 0.1, 10.0, type: {:f, 64})
          check_grads!(&Nx.unquote(fun)(&1), &(__MODULE__.unquote(grad_fun) / 1), t)
        end
      end
    end
  end
end
