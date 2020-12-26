defmodule Nx.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.GradHelpers
  doctest Nx.Defn.Grad

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

  # describe "dot rule" do
  #   defn dot_rule(t), do: Nx.tanh(Nx.tanh(Nx.dot(Nx.power(t, 2), Nx.power(t, 3))))
  #   defn grad_dot_rule(t), do: grad(t, dot_rule(t))

  #   test "computes gradient for scalars" do
  #     assert grad_product_rule(Nx.tensor(1.0)) == Nx.tensor(1.2343397629215758)

  #     for _ <- 1..100 do
  #       check_grads!(
  #         &product_rule/1,
  #         &grad_product_rule/1,
  #         Nx.random_uniform({}, 0.0, 1000.0, type: {:f, 64})
  #       )
  #     end
  #   end

  #   defn grad_dot_lhs_rule(x, y), do: grad(x, Nx.sum(Nx.dot(x, y)))

  #   test "computes gradient for tensors on lhs" do
  #     assert grad_dot_lhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
  #              Nx.tensor([[15.0], [15.0], [15.0]])

  #     assert grad_dot_lhs_rule(
  #              Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
  #              Nx.tensor([1.0, 2.0])
  #            ) ==
  #              Nx.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])

  #     assert grad_dot_lhs_rule(
  #              Nx.tensor([1.0, 2.0]),
  #              Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  #            ) ==
  #              Nx.tensor([6.0, 15.0])
  #   end

  #   defn grad_dot_rhs_rule(x, y), do: grad(y, Nx.sum(Nx.dot(x, y)))

  #   test "computes gradient for tensors on rhs" do
  #     assert grad_dot_rhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
  #              Nx.tensor([[6.0, 6.0, 6.0, 6.0, 6.0]])

  #     assert grad_dot_rhs_rule(
  #              Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
  #              Nx.tensor([1.0, 2.0])
  #            ) ==
  #              Nx.tensor([9.0, 12.0])

  #     assert grad_dot_rhs_rule(
  #              Nx.tensor([1.0, 2.0]),
  #              Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  #            ) ==
  #              Nx.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
  #   end

  #   defn grad_dot_both_rule(x), do: grad(x, Nx.sum(Nx.dot(Nx.power(x, 2), Nx.power(x, 3))))

  #   test "computes gradient for tensors on both sides" do
  #     assert grad_dot_both_rule(Nx.iota({3, 3, 3})) ==
  #              Nx.tensor([
  #                [
  #                  [0.0, 83430.0, 263_952.0],
  #                  [198_207.0, 410_616.0, 759_375.0],
  #                  [533_952.0, 884_142.0, 1_410_048.0]
  #                ],
  #                [
  #                  [873_828.0, 1_330_020.0, 1_997_028.0],
  #                  [1_460_592.0, 2_057_913.0, 2_905_308.0],
  #                  [2.268e6, 3_016_224.0, 4_053_888.0]
  #                ],
  #                [
  #                  [2_639_952.0, 3_468_906.0, 4.6224e6],
  #                  [3_724_623.0, 4_706_856.0, 6_052_887.0],
  #                  [5_121_792.0, 6_268_050.0, 7_817_472.0]
  #                ]
  #              ])

  #     assert grad_dot_both_rule(Nx.tensor([1, 2, 3])) == Nx.tensor([5.0, 80.0, 405.0])
  #   end

  #   defn grad_dot_dot_rule(x, w1, b1, w2, b2) do
  #     grad(
  #       x,
  #       x
  #       |> Nx.dot(w1)
  #       |> Nx.add(b1)
  #       |> Nx.dot(w2)
  #       |> Nx.add(b2)
  #       |> Nx.sum()
  #     ))
  #   end

  #   test "computes gradient with dot after dot" do
  #     assert grad_dot_dot_rule(
  #              Nx.iota({5, 4}),
  #              Nx.iota({4, 3}),
  #              Nx.iota({3}),
  #              Nx.iota({3, 2}),
  #              Nx.iota({2})
  #            ) ==
  #              Nx.tensor([
  #                [2.8414709848078967, 2.1310220466218284],
  #                [-0.8231163613806731, -2.909552551516233]
  #              ])
  #   end

  #   TODO: grad for outer, transpose, reshape
  #   defn grad_dot_transpose_rule(x), do: grad(x, Nx.sum(Nx.dot(x, Nx.transpose(x))))

  #   test "computes gradient with transpose" do
  #     assert grad_dot_transpose_rule(Nx.iota({2, 3})) ==
  #              Nx.tensor([1])
  #   end
  # end

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

  for fun <-
        [:cbrt, :cos, :exp, :expm1, :log, :log1p, :logistic] ++
          [:mean, :negate, :rsqrt, :sin, :sqrt, :sum, :tanh] do
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

  describe "broadcast" do
    defn grad_sum_broadcast(t), do: grad(t, Nx.sum(Nx.broadcast(t, {2, 2})))

    test "computes gradient" do
      assert grad_sum_broadcast(Nx.tensor([[0.0, 1.0], [2.0, 3.0]])) ==
               Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      assert grad_sum_broadcast(Nx.tensor([[0.0, 1.0]])) ==
               Nx.tensor([2.0, 2.0])

      assert grad_sum_broadcast(Nx.tensor([0.0, 1.0])) ==
               Nx.tensor([2.0, 2.0])

      assert grad_sum_broadcast(Nx.tensor(0.0)) ==
               Nx.tensor(4.0)
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
      assert grad_log_sum_0_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.18345697474330172, -0.33410075509515635, -0.3228698817445151],
                 [0.04586424368582543, -0.13364030203806254, -0.16143494087225754]
               ])

      assert grad_log_sum_1_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [-0.21916944995978982, -0.10958472497989491, -0.07305648331992994],
                 [0.01875804509762369, 0.015006436078098954, 0.012505363398415794]
               ])
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

    #   defn grad_reshape_mean_0_sum(t),
    #     do: grad(t, t |> Nx.reshape({3, 2}) |> Nx.mean(axes: [0]) |> Nx.sum())

    #   defn grad_reshape_mean_1_sum(t),
    #     do: grad(t, t |> Nx.reshape({3, 2}) |> Nx.mean(axes: [1]) |> Nx.sum())

    #   test "computes reshape + mean(axis) + sum" do
    #     assert grad_reshape_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
    #              Nx.tensor([
    #                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
    #                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    #              ])

    #     assert grad_reshape_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
    #              Nx.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    #   end

    #   defn grad_reshape_mean_0(t),
    #     do: grad(t, t |> Nx.reshape({6}) |> Nx.mean(axes: [0]))

    #   test "computes reshape + mean(axis)" do
    #     assert grad_reshape_mean_0(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
    #              Nx.tensor([
    #                [0.16666666666666666, 0.16666666666666666, 0.16666666666666666],
    #                [0.16666666666666666, 0.16666666666666666, 0.16666666666666666]
    #              ])
    #   end

    #   defn grad_transpose_mean_0_sum(t),
    #     do: grad(t, t |> Nx.transpose() |> Nx.mean(axes: [0]) |> Nx.sum())

    #   defn grad_transpose_mean_1_sum(t),
    #     do: grad(t, t |> Nx.transpose() |> Nx.mean(axes: [1]) |> Nx.sum())

    #   test "computes transpose + mean(axis) + sum" do
    #     assert grad_transpose_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
    #              Nx.tensor([
    #                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
    #                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    #              ])

    #     assert grad_transpose_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
    #              Nx.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    #   end
  end
end
