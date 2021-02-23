defmodule Nx.Defn.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.GradHelpers

  @iters 1..25

  describe "simple" do
    defn grad_itself(t), do: grad(t, t)
    defn grad_tensor(t), do: grad(t, Nx.tensor(1.0))
    defn grad_constant(t), do: grad(t, 1.0)
    defn grad_unrelated(t, a), do: grad(t, a)

    test "computes gradient for scalars" do
      assert grad_itself(Nx.tensor(1.0)) == Nx.tensor(1.0)
      assert grad_tensor(Nx.tensor(1.0)) == Nx.tensor(0.0)
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

  describe "value and grad" do
    defn value_and_grad(a, b) do
      expr = Nx.tanh(a) + Nx.power(b, 2)
      {expr, grad({a, b}, expr)}
    end

    test "computes value and grad" do
      assert value_and_grad(1, 2) ==
               {Nx.tensor(4.761594155955764), {Nx.tensor(0.41997434161402614), Nx.tensor(4.0)}}
    end
  end

  describe "cache" do
    defn subexpressions(x, y) do
      z = x * y
      Nx.sum(z - (z |> Nx.exp() |> Nx.sum(axes: [0]) |> Nx.log()))
    end

    defn grad_subexpressions(x, y), do: grad(x, subexpressions(x, y))

    test "considers current g" do
      assert grad_subexpressions(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3])) ==
               Nx.tensor([0.9990006807109719, 1.9598562710443452, -5.936786448699433])
    end
  end

  describe "addition rule" do
    defn addition_rule(t), do: Nx.tanh(Nx.tanh(Nx.add(Nx.power(t, 2), Nx.power(t, 3))))
    defn grad_addition_rule(t), do: grad(t, addition_rule(t))

    test "computes gradient of complex rules" do
      assert grad_addition_rule(Nx.tensor(1.0)) == Nx.tensor(0.1566267114813547)

      for _ <- @iters do
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

      for _ <- @iters do
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
      assert grad_sum_product_rule(Nx.tensor([[1.0, 2.0], [3.0, 4.0]])) ==
               Nx.tensor([[5.0, 80.0], [405.0, 1280.0]])
    end
  end

  describe "division rule" do
    defn division_rule(t), do: Nx.divide(Nx.tanh(t), t)
    defn grad_division_rule(t), do: grad(t, division_rule(t))

    test "computes gradient" do
      assert grad_division_rule(Nx.tensor(1.0)) == Nx.tensor(-0.3416198143417387)

      for _ <- @iters do
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
      for _ <- @iters do
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
      for _ <- @iters do
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

      for _ <- @iters do
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
      for _ <- @iters do
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
      for _ <- @iters do
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

      for _ <- @iters do
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
      assert grad_exp_rule(Nx.tensor(1.0)) == Nx.tensor(1.3704876904488987)

      for _ <- @iters do
        check_grads!(
          &exp_rule/1,
          &grad_exp_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "atan2 rule" do
    defn atan2_rule(t), do: Nx.atan2(Nx.tanh(t), t)
    defn grad_atan2_rule(t), do: grad(t, atan2_rule(t))

    test "computes gradient" do
      assert grad_atan2_rule(Nx.tensor(1.0)) == Nx.tensor(-0.21621156120382867)

      for _ <- @iters do
        check_grads!(
          &atan2_rule/1,
          &grad_atan2_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "dot rule" do
    defn grad_dot_lhs_rule(x, y), do: grad(x, Nx.sum(Nx.dot(x, y)))

    test "computes gradient for tensors on lhs" do
      assert grad_dot_lhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
               Nx.tensor([[15.0], [15.0], [15.0]])

      assert grad_dot_lhs_rule(
               Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
               Nx.tensor([1.0, 2.0])
             ) ==
               Nx.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])

      assert grad_dot_lhs_rule(
               Nx.tensor([1.0, 2.0]),
               Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
             ) ==
               Nx.tensor([6.0, 15.0])
    end

    defn grad_dot_rhs_rule(x, y), do: grad(y, Nx.sum(Nx.dot(x, y)))

    test "computes gradient for tensors on rhs" do
      assert grad_dot_rhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
               Nx.tensor([[6.0, 6.0, 6.0, 6.0, 6.0]])

      assert grad_dot_rhs_rule(
               Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
               Nx.tensor([1.0, 2.0])
             ) ==
               Nx.tensor([9.0, 12.0])

      assert grad_dot_rhs_rule(
               Nx.tensor([1.0, 2.0]),
               Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
             ) ==
               Nx.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    end

    defn grad_dot_both_rule(x), do: grad(x, Nx.sum(Nx.dot(Nx.power(x, 2), Nx.power(x, 3))))

    test "computes gradient for tensors on both sides" do
      assert grad_dot_both_rule(Nx.iota({3, 3, 3})) ==
               Nx.tensor([
                 [
                   [0.0, 83430.0, 263_952.0],
                   [198_207.0, 410_616.0, 759_375.0],
                   [533_952.0, 884_142.0, 1_410_048.0]
                 ],
                 [
                   [873_828.0, 1_330_020.0, 1_997_028.0],
                   [1_460_592.0, 2_057_913.0, 2_905_308.0],
                   [2.268e6, 3_016_224.0, 4_053_888.0]
                 ],
                 [
                   [2_639_952.0, 3_468_906.0, 4.6224e6],
                   [3_724_623.0, 4_706_856.0, 6_052_887.0],
                   [5_121_792.0, 6_268_050.0, 7_817_472.0]
                 ]
               ])

      assert grad_dot_both_rule(Nx.tensor([1, 2, 3])) == Nx.tensor([5.0, 80.0, 405.0])
    end

    defn grad_dot_dot_rule(x, w1, b1, w2, b2, labels) do
      grad(
        x,
        x
        |> Nx.dot(w1)
        |> Nx.add(b1)
        |> Nx.dot(w2)
        |> Nx.add(b2)
        |> Nx.multiply(labels)
        |> Nx.sum()
      )
    end

    test "computes gradient with dot after dot" do
      assert grad_dot_dot_rule(
               Nx.iota({5, 4}),
               Nx.iota({4, 3}),
               Nx.iota({3}),
               Nx.iota({3, 2}),
               Nx.iota({2}),
               Nx.iota({5, 2})
             ) ==
               Nx.tensor([
                 [13.0, 40.0, 67.0, 94.0],
                 [59.0, 176.0, 293.0, 410.0],
                 [105.0, 312.0, 519.0, 726.0],
                 [151.0, 448.0, 745.0, 1042.0],
                 [197.0, 584.0, 971.0, 1358.0]
               ])
    end

    defn grad_dot_implicit_bcast_rule(b1, w2, labels) do
      grad(
        b1,
        b1
        |> Nx.dot(w2)
        |> Nx.multiply(labels)
        |> Nx.sum()
      )
    end

    test "computes gradient with dot with implicit broadcast" do
      assert grad_dot_implicit_bcast_rule(
               Nx.iota({3}),
               Nx.iota({3, 2}),
               Nx.iota({5, 2})
             ) ==
               Nx.tensor([25.0, 115.0, 205.0])
    end
  end

  describe "outer rule" do
    defn grad_outer_lhs_rule(x, y), do: grad(x, Nx.sum(Nx.outer(x, y)))

    test "computes gradient for tensors on lhs" do
      assert grad_outer_lhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
               Nx.tensor([[15.0], [15.0], [15.0]])
    end

    defn grad_outer_rhs_rule(x, y), do: grad(y, Nx.sum(Nx.outer(x, y)))

    test "computes gradient for tensors on rhs" do
      assert grad_outer_rhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
               Nx.tensor([[6.0, 6.0, 6.0, 6.0, 6.0]])
    end

    defn grad_outer_both_rule(x), do: grad(x, Nx.sum(Nx.outer(Nx.power(x, 2), Nx.power(x, 3))))

    test "computes gradient for tensors on both sides" do
      assert grad_outer_both_rule(Nx.iota({3, 3, 3})) ==
               Nx.tensor([
                 [
                   [0.0, 265_005.0, 567_216.0],
                   [906_633.0, 1_283_256.0, 1_697_085.0],
                   [2_148_120.0, 2_636_361.0, 3_161_808.0]
                 ],
                 [
                   [3_724_461.0, 4_324_320.0, 4_961_385.0],
                   [5_635_656.0, 6_347_133.0, 7_095_816.0],
                   [7_881_705.0, 8.7048e6, 9_565_101.0]
                 ],
                 [
                   [10_462_608.0, 11_397_321.0, 12_369_240.0],
                   [13_378_365.0, 14_424_696.0, 15_508_233.0],
                   [16_628_976.0, 17_786_925.0, 18_982_080.0]
                 ]
               ])

      assert grad_outer_both_rule(Nx.tensor([1, 2, 3])) == Nx.tensor([114.0, 312.0, 594.0])
    end
  end

  describe "chain rule" do
    defn grad_tanh_exp(t), do: grad(t, Nx.tanh(Nx.exp(t)))

    test "computes gradient" do
      assert grad_tanh_exp(Nx.tensor(1.0)) == Nx.tensor(0.04693651986265914)

      for _ <- @iters do
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

      for _ <- @iters do
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

    defn grad_tuple_output(a), do: grad(a, {a + 1, a - 1})

    test "raises on tuple output" do
      assert_raise ArgumentError, ~r"expected a tensor or a numbe", fn ->
        grad_tuple_output(Nx.tensor(1.0))
      end
    end
  end

  for fun <-
        [:cbrt, :cos, :exp, :expm1, :log, :log1p, :logistic] ++
          [:mean, :negate, :rsqrt, :sin, :sqrt, :sum, :tanh] do
    describe "#{fun}" do
      grad_fun = :"grad_#{fun}"
      defn unquote(grad_fun)(t), do: grad(t, Nx.unquote(fun)(t))

      test "computes gradient" do
        for _ <- @iters do
          t = Nx.random_uniform({}, 0.1, 10.0, type: {:f, 64})
          check_grads!(&Nx.unquote(fun)(&1), &(__MODULE__.unquote(grad_fun) / 1), t)
        end
      end
    end
  end

  describe "tan" do
    defn grad_tan(t), do: grad(t, Nx.tan(t))

    test "computes gradient" do
      for _ <- @iters do
          # check_grads!/4 fails for values close to the asymptotes
          # of tan's gradient, so we select t to avoid them.
          multiplier = Nx.random_uniform({}, 0, 10, type: {:u, 32})
          offset = Nx.random_uniform({}, -1.5, 1.5, type: {:f, 64})
          t = 3.14159 |> Nx.multiply(multiplier) |> Nx.add(offset)
          check_grads!(&Nx.tan/1, &grad_tan/1, t)
        end
    end
  end

  describe "inverse trig family" do
    defn grad_asin(t), do: grad(t, Nx.asin(t))
    defn grad_acos(t), do: grad(t, Nx.acos(t))
    defn grad_atan(t), do: grad(t, Nx.atan(t))

    test "computes gradient of inverse trig functions" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -0.999, 0.999, type: {:f, 32})
        check_grads!(&Nx.asin/1, &grad_asin/1, t, eps: 0.1)
        check_grads!(&Nx.acos/1, &grad_acos/1, t, eps: 0.1)
        check_grads!(&Nx.atan/1, &grad_atan/1, t, eps: 0.1)
        check_grads!(&Nx.atan/1, &grad_atan/1, Nx.multiply(1000.0,t), eps: 0.1)
      end
    end
  end

  describe "hyperbolics" do
    defn grad_sinh(t), do: grad(t, Nx.sinh(t))
    defn grad_cosh(t), do: grad(t, Nx.cosh(t))

    test "computes gradient" do
      for _ <- @iters do
          t = Nx.random_uniform({}, -10, 10, type: {:f, 64})
          check_grads!(&Nx.sinh/1, &grad_sinh/1, t)
          check_grads!(&Nx.cosh/1, &grad_cosh/1, t)
        end
    end
  end

  describe "inverse hyperbolic functions" do
    defn grad_asinh(t), do: grad(t, Nx.asinh(t))
    defn grad_acosh(t), do: grad(t, Nx.acosh(t))
    defn grad_atanh(t), do: grad(t, Nx.atanh(t))

    test "computes gradient of inverse hyperbolic functions" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -100.0, 100.0, type: {:f, 64})
        check_grads!(&Nx.asinh/1, &grad_asinh/1, t, eps: 0.1)
      end

      for _ <- @iters do
        t = Nx.random_uniform({}, 1.01, 100.0, type: {:f, 64})
        check_grads!(&Nx.acosh/1, &grad_acosh/1, t, eps: 0.1)
      end

      for _ <- @iters do
        t = Nx.random_uniform({}, -0.999, 0.999, type: {:f, 64})
        check_grads!(&Nx.atanh/1, &grad_atanh/1, t, eps: 0.1)
      end
    end
  end

  describe "erf_inv" do
    defn grad_erf_inv(t), do: grad(t, Nx.erf_inv(t))

    test "computes gradient close to 0.0" do
      for _ <- @iters do
        t = Nx.random_uniform({}, 0.0, 0.9, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, eps: 1.0e-4)
      end
    end

    test "computes gradient between 0.9 and 0.95" do
      for _ <- @iters do
        t = Nx.random_uniform({}, 0.9, 0.95, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, eps: 1.0e-3)
      end
    end

    test "computes gradient between 0.95 and 0.98" do
      for _ <- @iters do
        t = Nx.random_uniform({}, 0.95, 0.98, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, eps: 0.00004)
      end
    end

    test "computes gradient approaching 1.0 but is sharply curved" do
      #check_grads! does not work near 1 due to sharp curve between close x's
      coords = [
        {0.9, 3.43},
        {0.98, 13.26},
        {0.99, 24.45},
        {0.991, 26.85},
        {0.992, 29.84},
        {0.993, 33.64},
        {0.994, 38.64},
        {0.995, 45.56},
        {0.999, 198.94},
      ]
      for {x, y} <- coords do
        assert_in_delta(Nx.to_scalar(grad_erf_inv(x)), y, 0.01)
      end
    end
  end

  describe "broadcast" do
    defn grad_sum_broadcast(t), do: grad(t, Nx.sum(Nx.broadcast(t, {3, 2, 2})))

    test "computes gradient" do
      assert grad_sum_broadcast(Nx.iota({3, 2, 2})) == Nx.broadcast(1.0, {3, 2, 2})
      assert grad_sum_broadcast(Nx.iota({1, 2, 2})) == Nx.broadcast(3.0, {1, 2, 2})
      assert grad_sum_broadcast(Nx.iota({3, 1, 2})) == Nx.broadcast(2.0, {3, 1, 2})
      assert grad_sum_broadcast(Nx.iota({3, 2, 1})) == Nx.broadcast(2.0, {3, 2, 1})
      assert grad_sum_broadcast(Nx.iota({3, 1, 1})) == Nx.broadcast(4.0, {3, 1, 1})
      assert grad_sum_broadcast(Nx.iota({1, 1, 1})) == Nx.broadcast(12.0, {1, 1, 1})

      assert grad_sum_broadcast(Nx.iota({2, 2})) == Nx.broadcast(3.0, {2, 2})
      assert grad_sum_broadcast(Nx.iota({1, 2})) == Nx.broadcast(6.0, {1, 2})
      assert grad_sum_broadcast(Nx.iota({2, 1})) == Nx.broadcast(6.0, {2, 1})

      assert grad_sum_broadcast(Nx.iota({2})) == Nx.broadcast(6.0, {2})
      assert grad_sum_broadcast(Nx.iota({})) == Nx.broadcast(12.0, {})
    end
  end

  describe "squeeze" do
    defn grad_sum_squeeze_broadcast(t),
      do: grad(t, Nx.sum(Nx.squeeze(Nx.broadcast(t, {3, 2, 2}))))

    test "computes gradient" do
      assert grad_sum_squeeze_broadcast(Nx.iota({3, 2, 2})) == Nx.broadcast(1.0, {3, 2, 2})
      assert grad_sum_squeeze_broadcast(Nx.iota({1, 2, 2})) == Nx.broadcast(3.0, {1, 2, 2})
      assert grad_sum_squeeze_broadcast(Nx.iota({1, 1, 2})) == Nx.broadcast(6.0, {1, 1, 2})
      assert grad_sum_squeeze_broadcast(Nx.iota({1, 1, 1})) == Nx.broadcast(12.0, {1, 1, 1})

      assert grad_sum_squeeze_broadcast(Nx.iota({2, 2})) == Nx.broadcast(3.0, {2, 2})
      assert grad_sum_squeeze_broadcast(Nx.iota({1, 2})) == Nx.broadcast(6.0, {1, 2})
      assert grad_sum_squeeze_broadcast(Nx.iota({1, 1})) == Nx.broadcast(12.0, {1, 1})

      assert grad_sum_squeeze_broadcast(Nx.iota({2})) == Nx.broadcast(6.0, {2})
      assert grad_sum_squeeze_broadcast(Nx.iota({1})) == Nx.broadcast(12.0, {1})
      assert grad_sum_squeeze_broadcast(Nx.iota({})) == Nx.broadcast(12.0, {})
    end
  end

  describe "pad" do
    defn grad_sum_pad(t), do: grad(t, Nx.sum(Nx.pad(t, 2.0, [{-1, 1, 0}, {1, 1, 0}])))
    defn grad_interior_pad(t), do: grad(t, Nx.sum(Nx.pad(t, 2.0, [{0, 0, 1}, {0, 0, 1}])))
    defn grad_lots_of_pad(t), do: grad(t, Nx.sum(Nx.pad(t, 2.0, [{-2, 1, 4}, {1, 3, 2}])))

    defn grad_pad_fun(t),
      do: grad(t, Nx.mean(Nx.pad(t, Nx.mean(Nx.cos(t)), [{-2, 1, 4}, {1, 3, 2}])))

    test "computes gradient" do
      assert grad_sum_pad(Nx.tensor([[1.0, 2.0], [1.0, 2.0]])) ==
               Nx.tensor([[0.0, 0.0], [1.0, 1.0]])
    end

    test "computes gradient with interior pad" do
      assert grad_interior_pad(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    end

    test "computes gradient with diverse pad" do
      assert grad_lots_of_pad(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    end

    test "computes with pad value from tensor" do
      assert grad_pad_fun(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([
                 [-0.13259542790912313, -0.1432832308937438, -0.022237092179130596],
                 [0.1374355447151887, 0.1692850372196461, 0.06221092698892166]
               ])
    end
  end

  describe "slice" do
    defn grad_mean_slice(t), do: grad(t, Nx.mean(Nx.slice(t, [0, 1], [1, 2], strides: [1, 2])))
    defn grad_sum_slice(t), do: grad(t, Nx.sum(Nx.slice(t, [1, 0], [1, 2], strides: [1, 1])))

    defn grad_sum_pad_slice(t) do
      grad(
        t,
        Nx.slice(t, [1, 0], [1, 2], strides: [1, 1])
        |> Nx.pad(Nx.mean(Nx.sin(t)), [{2, 1, 2}, {-1, 2, 0}])
        |> Nx.sum()
      )
    end

    test "computes gradient" do
      assert grad_mean_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

      assert grad_sum_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

      lhs = grad_sum_pad_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

      rhs =
        Nx.tensor([
          [0.9905542274249228, -0.7629358670030943, -1.8149862437674833],
          [-1.1983466382499552, 1.520047340015915, 1.7603121921923377]
        ])

      compare_tensors!(lhs, rhs)
    end
  end

  describe "reverse" do
    defn grad_sum_reverse_exp(t), do: grad(t, Nx.sum(Nx.reverse(Nx.exp(t))))
    defn grad_sum_exp_reverse(t), do: grad(t, Nx.sum(Nx.exp(Nx.reverse(t))))

    test "computes gradient" do
      assert grad_sum_exp_reverse(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([
                 [2.718281828459045, 7.38905609893065, 20.085536923187668],
                 [54.598150033144236, 148.4131591025766, 403.4287934927351]
               ])

      assert grad_sum_reverse_exp(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([
                 [2.718281828459045, 7.38905609893065, 20.085536923187668],
                 [54.598150033144236, 148.4131591025766, 403.4287934927351]
               ])
    end
  end

  describe "abs" do
    defn abs_scalar(t), do: Nx.abs(t)
    defn grad_abs_scalar(t), do: grad(t, Nx.abs(t))
    defn grad_abs(t), do: grad(t, Nx.sum(Nx.abs(t)))

    test "computes gradient with scalars" do
      for _ <- @iters do
        check_grads!(
          &abs_scalar/1,
          &grad_abs_scalar/1,
          Nx.random_uniform({}, 0.0, 1000.0, type: {:f, 64})
        )
      end
    end

    test "computes gradient with tensors" do
      assert grad_abs(Nx.tensor([[1.0, 2.0], [3.0, 4.0]])) == Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      assert grad_abs(Nx.tensor([[-1.0, 2.0], [-3.0, 4.0]])) ==
               Nx.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    end
  end

  describe "max" do
    defn grad_max(t), do: grad(t, Nx.sum(Nx.max(Nx.power(t, 2), Nx.power(t, 3))))

    test "computes gradient with tensors" do
      assert grad_max(Nx.tensor([[1.0], [2.0], [3.0]])) == Nx.tensor([[2.5], [12.0], [27.0]])

      assert grad_max(Nx.tensor([[1.25, 2.5, 2.75], [1.0, 4.0, 6.0], [2.0, 3.0, 2.0]])) ==
               Nx.tensor([[4.6875, 18.75, 22.6875], [2.5, 48.0, 108.0], [12.0, 27.0, 12.0]])
    end
  end

  describe "min" do
    defn grad_min(t), do: grad(t, Nx.sum(Nx.min(Nx.power(t, 2), Nx.power(t, 3))))

    test "computes gradient with tensors" do
      assert grad_min(Nx.tensor([[1.0], [2.0], [3.0]])) == Nx.tensor([[2.5], [4.0], [6.0]])

      assert grad_min(Nx.tensor([[1.25, 2.5, 2.75], [1.0, 4.0, 6.0], [2.0, 3.0, 2.0]])) ==
               Nx.tensor([[2.5, 5.0, 5.5], [2.5, 8.0, 12.0], [4.0, 6.0, 4.0]])
    end
  end

  describe "select rule" do
    defn grad_sum_select(t),
      do: grad(t, Nx.sum(Nx.select(Nx.greater(t, 0.0), Nx.exp(t), Nx.cos(t))))

    defn grad_max_select(t),
      do: grad(t, Nx.reduce_max(Nx.select(Nx.greater(t, 0.0), Nx.exp(t), Nx.cos(t))))

    test "computes gradient with sum+select" do
      assert grad_sum_select(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]])) ==
               Nx.tensor([
                 [
                   0.9092974268256817,
                   2.718281828459045,
                   0.0,
                   20.085536923187668,
                   0.1411200080598672
                 ],
                 [2.718281828459045, 7.38905609893065, 0.0, 148.4131591025766, 0.8414709848078965]
               ])
    end

    test "computes the gradient with max+select" do
      assert grad_max_select(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 148.4131591025766, 0.0]])
    end
  end

  describe "as_type" do
    defn grad_as_type(t), do: grad(t, Nx.sum(Nx.as_type(t, {:f, 32})))

    test "passes through" do
      assert grad_as_type(Nx.tensor([1, 2, 3])) == Nx.tensor([1.0, 1.0, 1.0])
    end
  end

  describe "if" do
    defn grad_if(t), do: grad(t, if(t + 1, do: Nx.power(t, 2), else: Nx.power(t, 3)))

    defn grad_sum_if(t),
      do: grad(t, Nx.sum(if(Nx.all?(t), do: Nx.power(t, 2), else: Nx.power(t, 3))))

    defn grad_if_sum(t),
      do: grad(t, if(Nx.all?(t), do: Nx.sum(Nx.power(t, 2)), else: Nx.sum(Nx.power(t, 3))))

    defn grad_if_tuple(t) do
      {{a, b}, c} =
        if t + 1 do
          {{Nx.power(t, 2), Nx.power(t, 3)}, Nx.power(t, 4)}
        else
          {{Nx.power(t, 4), Nx.power(t, 3)}, Nx.power(t, 2)}
        end

      grad(t, a * b + c)
    end

    test "computes gradient" do
      assert grad_if(Nx.tensor(1)) == Nx.tensor(2.0)
      assert grad_if(Nx.tensor(-1)) == Nx.tensor(3.0)
    end

    test "computes gradient with sum" do
      assert grad_sum_if(Nx.tensor([1, 2, 3])) == Nx.tensor([2.0, 4.0, 6.0])
      assert grad_sum_if(Nx.tensor([-1, 0, 1])) == Nx.tensor([3.0, 0.0, 3.0])

      assert grad_if_sum(Nx.tensor([1, 2, 3])) == Nx.tensor([2.0, 4.0, 6.0])
      assert grad_if_sum(Nx.tensor([-1, 0, 1])) == Nx.tensor([3.0, 0.0, 3.0])
    end

    test "computes gradient with tuple" do
      assert grad_if_tuple(Nx.tensor(1)) == Nx.tensor(9.0)
      assert grad_if_tuple(Nx.tensor(-1)) == Nx.tensor(5.0)
    end
  end

  describe "axes" do
    defn grad_sum_full(t), do: grad(t, Nx.sum(t))
    defn grad_mean_full(t), do: grad(t, Nx.mean(t))

    test "computes gradient in full" do
      assert grad_sum_full(Nx.tensor([[1, 2], [3, 4]])) ==
               Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      assert grad_mean_full(Nx.tensor([[1, 2], [3, 4]])) ==
               Nx.tensor([[0.25, 0.25], [0.25, 0.25]])
    end

    defn grad_log_sum_0_sin_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.sum(axes: [0]) |> Nx.sin() |> Nx.sum())

    defn grad_log_sum_1_sin_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.sum(axes: [1]) |> Nx.sin() |> Nx.sum())

    test "computes log + sum(axis) + sin + sum" do
      lhs = grad_log_sum_0_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [0.18345697474330172, -0.33410075509515635, -0.3228698817445151],
          [0.04586424368582543, -0.13364030203806254, -0.16143494087225754]
        ])

      compare_tensors!(lhs, rhs)

      lhs = grad_log_sum_1_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [-0.21916944995978982, -0.10958472497989491, -0.07305648331992994],
          [0.01875804509762369, 0.015006436078098952, 0.012505363398415794]
        ])

      compare_tensors!(lhs, rhs)
    end

    defn grad_log_sum_keep_sin_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.sum(axes: [1], keep_axes: true) |> Nx.sin() |> Nx.sum())

    test "computes log + sum(keep_axes) + sin + sum" do
      lhs = grad_log_sum_keep_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [-0.21916944995978982, -0.10958472497989491, -0.07305648331992994],
          [0.01875804509762369, 0.015006436078098952, 0.012505363398415794]
        ])

      compare_tensors!(lhs, rhs)
    end

    defn grad_sum_0_mean(t), do: grad(t, t |> Nx.sum(axes: [0]) |> Nx.mean())
    defn grad_sum_1_mean(t), do: grad(t, t |> Nx.sum(axes: [1]) |> Nx.mean())

    test "computes sum(axis) + mean" do
      assert grad_sum_0_mean(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
               ])

      assert grad_sum_1_mean(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    end

    defn grad_mean_0_sum(t), do: grad(t, t |> Nx.mean(axes: [0]) |> Nx.sum())
    defn grad_mean_1_sum(t), do: grad(t, t |> Nx.mean(axes: [1]) |> Nx.sum())

    test "computes mean(axis) + sum" do
      assert grad_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

      assert grad_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
               ])
    end

    defn grad_reshape_mean_0_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.reshape({3, 2}) |> Nx.mean(axes: [0]) |> Nx.sum())

    defn grad_reshape_mean_1_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.reshape({3, 2}) |> Nx.mean(axes: [1]) |> Nx.sum())

    test "computes log + reshape + mean(axis) + sum" do
      assert grad_reshape_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.16666666666666666, 0.1111111111111111],
                 [0.08333333333333333, 0.06666666666666667, 0.05555555555555555]
               ])

      assert grad_reshape_mean_0_sum(Nx.tensor([1, 2, 3, 4, 5, 6])) ==
               Nx.tensor([
                 0.3333333333333333,
                 0.16666666666666666,
                 0.1111111111111111,
                 0.08333333333333333,
                 0.06666666666666667,
                 0.05555555555555555
               ])

      assert grad_reshape_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.25, 0.16666666666666666], [0.125, 0.1, 0.08333333333333333]])
    end

    defn grad_transpose_mean_0_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.transpose() |> Nx.mean(axes: [0]) |> Nx.sum())

    defn grad_transpose_mean_1_sum(t),
      do: grad(t, t |> Nx.log() |> Nx.transpose() |> Nx.mean(axes: [1]) |> Nx.sum())

    test "computes log + transpose + mean(axis) + sum" do
      assert grad_transpose_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.16666666666666666, 0.1111111111111111],
                 [0.08333333333333333, 0.06666666666666667, 0.05555555555555555]
               ])

      assert grad_transpose_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.25, 0.16666666666666666], [0.125, 0.1, 0.08333333333333333]])
    end
  end

  describe "clip" do
    defn grad_sum_clip(t), do: grad(t, Nx.sum(Nx.clip(t, Nx.tensor(1.0), Nx.tensor(4.0))))

    test "computes gradient with sum" do
      assert grad_sum_clip(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    end
  end

  describe "reduce_max rule" do
    defn grad_reduce_max(t), do: grad(t, Nx.reduce_max(Nx.cos(Nx.exp(t))))
    defn grad_sum_reduce_max(t), do: grad(t, Nx.sum(Nx.reduce_max(t, axes: [1])))
    defn grad_sum_reduce_max_cos(t), do: grad(t, Nx.sum(Nx.reduce_max(Nx.cos(t), axes: [1])))
    defn grad_reduce_max_sum(t), do: grad(t, Nx.reduce_max(Nx.sum(t, axes: [1])))
    defn grad_reduce_max_min(t), do: grad(t, Nx.reduce_max(Nx.reduce_min(t, axes: [0])))

    defn grad_reduce_max_min_sum(t),
      do: grad(t, Nx.reduce_max(Nx.reduce_min(Nx.sum(t, axes: [1]), axes: [0])))

    test "computes gradient" do
      lhs = grad_reduce_max(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))
      rhs = Nx.tensor([[0.0, -3.302372203078941, 0.0, 0.0], [-3.302372203078941, 0.0, 0.0, 0.0]])
      compare_tensors!(lhs, rhs)
    end

    test "computes gradient with sum" do
      lhs = grad_sum_reduce_max(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))
      rhs = Nx.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
      compare_tensors!(lhs, rhs)
    end

    test "computes gradient with sum+cos" do
      lhs = grad_sum_reduce_max_cos(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))

      rhs =
        Nx.tensor([
          [-0.8414709848078965, 0.0, 0.0, 0.0],
          [0.0, -0.42073549240394825, 0.0, -0.42073549240394825]
        ])

      compare_tensors!(lhs, rhs)
    end

    test "computes gradient with max+sum" do
      assert grad_reduce_max_sum(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]])) ==
               Nx.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    end

    test "computes gradient with max+min" do
      assert grad_reduce_max_min(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0]])
    end

    test "computes the gradient with max+min+sum" do
      lhs =
        grad_reduce_max_min_sum(
          Nx.tensor([[[1.0, 0.0, 2.0], [3.0, 4.0, 2.0]], [[5.0, 2.0, 3.0], [4.0, 2.0, 1.0]]])
        )

      rhs =
        Nx.tensor([
          [[0.33333334, 0.16666667, 0.16666667], [0.33333334, 0.16666667, 0.16666667]],
          [[0.0, 0.16666667, 0.16666667], [0.0, 0.16666667, 0.16666667]]
        ])

      compare_tensors!(lhs, rhs)
    end
  end

  describe "reduce_min rule" do
    defn grad_reduce_min(t), do: grad(t, Nx.reduce_min(Nx.cos(Nx.exp(t))))
    defn grad_sum_reduce_min(t), do: grad(t, Nx.sum(Nx.reduce_min(t, axes: [1])))
    defn grad_sum_reduce_min_cos(t), do: grad(t, Nx.sum(Nx.reduce_min(Nx.cos(t), axes: [1])))
    defn grad_reduce_min_sum(t), do: grad(t, Nx.reduce_min(Nx.sum(t, axes: [1])))
    defn grad_reduce_min_max(t), do: grad(t, Nx.reduce_min(Nx.reduce_max(t, axes: [0])))

    defn grad_reduce_min_max_sum(t),
      do: grad(t, Nx.reduce_min(Nx.reduce_max(Nx.sum(t, axes: [1]), axes: [0])))

    test "computes gradient" do
      lhs = grad_reduce_min(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))

      rhs =
        Nx.tensor([
          [-0.37220643914833773, 0.0, 0.0, 0.0],
          [0.0, -0.37220643914833773, 0.0, -0.37220643914833773]
        ])

      compare_tensors!(lhs, rhs)
    end

    test "computes gradient with sum" do
      lhs = grad_sum_reduce_min(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))
      rhs = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.5]])
      compare_tensors!(lhs, rhs)
    end

    test "computes gradient with sum+cos" do
      lhs = grad_sum_reduce_min_cos(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))

      rhs =
        Nx.tensor([[0.0, 0.0, -0.1411200080598672, 0.0], [0.0, 0.0, -0.1411200080598672, 0.0]])

      compare_tensors!(lhs, rhs)
    end

    test "computes gradient with min+sum" do
      assert grad_reduce_min_sum(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    end

    test "computes gradient with min+max" do
      assert grad_reduce_min_max(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]])) ==
               Nx.tensor([[0.0, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    end

    test "computes the gradient with min+max+sum" do
      assert grad_reduce_min_max_sum(
               Nx.tensor([[[1.0, 0.0, 2.0], [3.0, 4.0, 2.0]], [[5.0, 2.0, 3.0], [4.0, 2.0, 1.0]]])
             ) ==
               Nx.tensor([
                 [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
                 [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]]
               ])
    end
  end

  describe "not implemented" do
    defn grad_reduce(t), do: grad(t, Nx.reduce(t, 0, fn x, y -> x + y end))

    test "raises on reduce" do
      assert_raise ArgumentError, ~r"cannot compute gradient for Nx.reduce/4", fn ->
        grad_reduce(3)
      end
    end

    defn grad_quotient(t), do: grad(t, Nx.quotient(t, 2))

    test "raises on quotient" do
      assert_raise ArgumentError, ~r"cannot compute gradient for Nx.quotient/2", fn ->
        grad_quotient(2)
      end
    end
  end

  # We need to round the floats because of imprecision between platforms
  defp compare_tensors!(
         %{type: {:f, size}, data: %{state: left_data} = lhs} = left,
         %{data: %{state: right_data} = rhs} = right
       ) do
    left_data = for <<x::float-size(size)-native <- left_data>>, do: Float.round(x, 5)
    right_data = for <<x::float-size(size)-native <- right_data>>, do: Float.round(x, 5)

    assert %{left | data: %{lhs | state: left_data}} == %{
             right
             | data: %{rhs | state: right_data}
           }
  end

  defp compare_tensors!(left, right) do
    assert left == right
  end
end
