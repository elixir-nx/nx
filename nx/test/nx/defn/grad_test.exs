defmodule Nx.Defn.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Helpers
  import Nx, only: :sigils

  @iters 1..25
  @types [{:f, 64}, {:c, 128}]

  defp random_uniform(min, max, opts) do
    rand =
      if Nx.Type.float?(opts[:type] || {:f, 32}) do
        :rand.uniform() * (max - min) + min
      else
        round(:rand.uniform() * (max - min) + min)
      end

    Nx.tensor(rand, opts)
  end

  describe "simple" do
    defn grad_itself(t), do: grad(t, fn t -> t end)
    defn grad_tensor(t), do: grad(t, fn _t -> Nx.tensor(1.0) end)
    defn grad_constant(t), do: grad(t, fn _t -> 10 end)
    defn grad_unrelated(t, a), do: grad(t, fn _t -> a end)
    defn grad_invalid(_t), do: grad(:invalid, fn t -> t end)
    defn grad_old_node(t), do: grad(t, fn _t -> t end)

    test "computes gradient for scalars" do
      assert grad_itself(Nx.tensor(1.0)) == Nx.tensor(1.0)
      assert grad_tensor(Nx.tensor(1.0)) == Nx.tensor(0.0)
      assert grad_constant(Nx.tensor(1.0)) == Nx.tensor(0.0)
      assert grad_unrelated(Nx.tensor(1.0), Nx.tensor(2.0)) == Nx.tensor(0.0)
      assert grad_old_node(Nx.tensor(1.0)) == Nx.tensor(0.0)
    end

    test "computes gradient for tensors" do
      assert grad_constant(Nx.tensor([1.0, 2.0, 3.0])) ==
               Nx.tensor([0.0, 0.0, 0.0])

      assert grad_unrelated(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor(2.0)) ==
               Nx.tensor([0.0, 0.0, 0.0])
    end

    test "computes gradient for multidimensional" do
      assert grad_itself(Nx.tensor([1.0, 2.0])) == Nx.tensor([1.0, 1.0])
      assert grad_tensor(Nx.tensor([1.0, 2.0])) == Nx.tensor([0.0, 0.0])
      assert grad_constant(Nx.tensor([1.0, 2.0])) == Nx.tensor([0.0, 0.0])
      assert grad_unrelated(Nx.tensor([1.0, 2.0]), Nx.tensor([2.0, 3.0])) == Nx.tensor([0.0, 0.0])
      assert grad_old_node(Nx.tensor([1.0, 2.0])) == Nx.tensor([0.0, 0.0])
    end

    test "raises on invalid" do
      assert_raise Protocol.UndefinedError, fn -> grad_invalid(Nx.tensor(1)) end
    end
  end

  describe "value and grad" do
    defn vg(a, b) do
      value_and_grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.pow(b, 2) end)
    end

    test "computes value and grad" do
      assert vg(1, 2) ==
               {Nx.tensor(4.761594155955764, type: {:f, 32}),
                {Nx.tensor(0.41997434161402614), Nx.tensor(4.0)}}
    end
  end

  describe "tokens" do
    defn grad_token(t), do: value_and_grad(t, fn t -> hook(Nx.pow(t, 2), :grad) end)

    test "computes grad with token" do
      parent = self()

      fun = Nx.Defn.jit(&grad_token/1, hooks: %{grad: &send(parent, {:hook, &1})})
      assert fun.(Nx.tensor(3)) == {Nx.tensor(9), Nx.tensor(6.0)}

      assert_receive {:hook, tensor}
      assert tensor == Nx.tensor(9)
    end

    defn token_grad(t), do: hook(grad(t, &Nx.pow(&1, 2)), :grad)

    test "computes token with grad" do
      parent = self()

      fun = Nx.Defn.jit(&token_grad/1, hooks: %{grad: &send(parent, {:hook, &1})})
      assert fun.(Nx.tensor(3)) == Nx.tensor(6.0)

      assert_receive {:hook, tensor}
      assert tensor == Nx.tensor(6.0)
    end
  end

  describe "metadata" do
    defn stop_grad_meta(t),
      do: grad(t, &stop_grad(Nx.exp(&1)))

    defn stop_grad_tuple_meta(a, b),
      do: grad({a, b}, fn {a, b} -> stop_grad(Nx.exp(a) + Nx.exp(b)) end)

    test "stops computing gradient" do
      assert stop_grad_meta(Nx.tensor(1)) == Nx.tensor(0.0)
      assert stop_grad_tuple_meta(Nx.tensor(1), Nx.tensor(1)) == {Nx.tensor(0.0), Nx.tensor(0.0)}
    end

    defn custom_grad_meta(t) do
      cos = grad(t, &Nx.cos/1)

      custom_cos =
        grad(t, fn t ->
          custom_grad(Nx.cos(t), [t], fn g ->
            [g * -Nx.sin(t)]
          end)
        end)

      {cos, custom_cos}
    end

    test "computes custom grad" do
      assert {x, x} = custom_grad_meta(Nx.tensor(1))
    end

    defn random_meta(t), do: grad(t, fn t -> t |> Nx.exp() |> random_meta_transform() end)

    deftransformp random_meta_transform(t) do
      Nx.Defn.Expr.metadata(t, %{oops: true})
    end

    test "ignores unknown metadata" do
      assert random_meta(Nx.tensor(1)) == Nx.exp(1)
    end
  end

  describe "cache" do
    defn subexpressions(x, y) do
      z = x * y
      Nx.sum(z - (z |> Nx.exp() |> Nx.sum(axes: [0]) |> Nx.log()))
    end

    defn grad_subexpressions(x, y), do: grad(x, &subexpressions(&1, y))

    test "considers current g" do
      assert grad_subexpressions(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3])) ==
               Nx.tensor([0.9990006807109719, 1.9598562710443452, -5.936786448699433])
    end
  end

  describe "addition rule" do
    defn addition_rule(t), do: Nx.tanh(Nx.tanh(Nx.add(Nx.pow(t, 2), Nx.pow(t, 3))))
    defn grad_addition_rule(t), do: grad(t, &addition_rule/1)

    test "computes gradient of compound rules" do
      assert grad_addition_rule(Nx.tensor(1.0)) == Nx.tensor(0.15662670135498047)

      for _ <- @iters, type <- @types do
        check_grads!(
          &addition_rule/1,
          &grad_addition_rule/1,
          random_uniform(-5.0, 5.0, type: type)
        )
      end
    end
  end

  describe "product rule" do
    defn product_rule(t), do: Nx.tanh(Nx.tanh(Nx.multiply(Nx.pow(t, 2), Nx.pow(t, 3))))
    defn grad_product_rule(t), do: grad(t, &product_rule/1)

    test "computes gradient for scalars" do
      assert grad_product_rule(Nx.tensor(1.0)) == Nx.tensor(1.2343397629215758)

      for _ <- @iters, type <- @types do
        check_grads!(
          &product_rule/1,
          &grad_product_rule/1,
          random_uniform(-0.5, 0.5, type: type)
        )
      end
    end

    defn sum_product_rule(t), do: Nx.sum(Nx.multiply(Nx.pow(t, 2), Nx.pow(t, 3)))
    defn grad_sum_product_rule(t), do: grad(t, &sum_product_rule/1)

    test "computes gradient for tensors" do
      assert grad_sum_product_rule(Nx.tensor([[1.0, 2.0], [3.0, 4.0]])) ==
               Nx.tensor([[5.0, 80.0], [405.0, 1280.0]])
    end
  end

  describe "division rule" do
    defn division_rule(t), do: Nx.divide(Nx.tanh(t), t)
    defn grad_division_rule(t), do: grad(t, &division_rule/1)

    test "computes gradient" do
      assert grad_division_rule(Nx.tensor(1.0)) == Nx.tensor(-0.3416198492050171)

      for _ <- @iters, type <- @types do
        check_grads!(
          &division_rule/1,
          &grad_division_rule/1,
          random_uniform(0.0, 10.0, type: type)
        )
      end
    end

    defn division_num_rule(t), do: Nx.divide(Nx.tanh(t), 2)
    defn grad_division_num_rule(t), do: grad(t, &division_num_rule/1)

    test "computes gradient for constant denominator" do
      for _ <- @iters do
        check_grads!(
          &division_num_rule/1,
          &grad_division_num_rule/1,
          random_uniform(0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn division_den_rule(t), do: Nx.divide(2, Nx.exp(t))
    defn grad_division_den_rule(t), do: grad(t, &division_den_rule/1)

    test "computes gradient for constant numerator" do
      for _ <- @iters do
        check_grads!(
          &division_den_rule/1,
          &grad_division_den_rule/1,
          random_uniform(0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "remainder rule" do
    defn remainder_rule(t), do: Nx.remainder(Nx.tanh(t), t)
    defn grad_remainder_rule(t), do: grad(t, &remainder_rule/1)

    test "computes gradient" do
      assert grad_remainder_rule(Nx.tensor(1.0)) == Nx.tensor(0.41997434161402614)

      for _ <- @iters do
        check_grads!(
          &remainder_rule/1,
          &grad_remainder_rule/1,
          random_uniform(0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn remainder_num_rule(t), do: Nx.remainder(Nx.tanh(t), 2)
    defn grad_remainder_num_rule(t), do: grad(t, &remainder_num_rule/1)

    test "computes gradient for constant denominator" do
      for _ <- @iters do
        check_grads!(
          &remainder_num_rule/1,
          &grad_remainder_num_rule/1,
          random_uniform(0.0, 10.0, type: {:f, 64})
        )
      end
    end

    defn remainder_den_rule(t), do: Nx.remainder(2, Nx.exp(t))
    defn grad_remainder_den_rule(t), do: grad(t, &remainder_den_rule/1)

    test "computes gradient for constant numerator" do
      for _ <- @iters do
        check_grads!(
          &remainder_den_rule/1,
          &grad_remainder_den_rule/1,
          random_uniform(0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "power rule" do
    defn power_rule(t), do: Nx.pow(t, 3)
    defn grad_power_rule(t), do: grad(t, &power_rule/1)

    test "computes gradient" do
      assert grad_power_rule(Nx.tensor(5.0)) == Nx.tensor(75.0)

      for _ <- @iters, type <- @types do
        check_grads!(
          &power_rule/1,
          &grad_power_rule/1,
          random_uniform(0.0, 10.0, type: type)
        )
      end
    end
  end

  describe "exponential rule" do
    defn exp_rule(t), do: Nx.add(Nx.pow(Nx.tanh(t), 2), Nx.pow(Nx.tanh(t), 3))
    defn grad_exp_rule(t), do: grad(t, &exp_rule/1)

    test "computes gradient" do
      assert grad_exp_rule(Nx.tensor(1.0)) == Nx.tensor(1.3704876904488987)

      for _ <- @iters, type <- @types do
        check_grads!(
          &exp_rule/1,
          &grad_exp_rule/1,
          random_uniform(0.0, 10.0, type: type)
        )
      end
    end
  end

  describe "atan2 rule" do
    defn atan2_rule(t), do: Nx.atan2(Nx.tanh(t), t)
    defn grad_atan2_rule(t), do: grad(t, &atan2_rule/1)

    test "computes gradient" do
      assert grad_atan2_rule(Nx.tensor(1.0)) == Nx.tensor(-0.21621156120382867)

      for _ <- @iters do
        check_grads!(
          &atan2_rule/1,
          &grad_atan2_rule/1,
          random_uniform(0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "dot rule" do
    defn grad_dot_lhs_rule(x, y), do: grad(x, &Nx.sum(Nx.dot(&1, y)))

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

      assert grad_dot_lhs_rule(
               Nx.tensor([Complex.new(0, 3), -2, Complex.new(0, 1)]),
               Nx.tensor([1.0, Complex.new(0, 2), 3.0])
             ) ==
               Nx.tensor([1, Complex.new(0, 2), 3])
    end

    defn grad_dot_rhs_rule(x, y), do: grad(y, &Nx.sum(Nx.dot(x, &1)))

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

    defn grad_dot_both_rule(x), do: grad(x, &Nx.sum(Nx.dot(Nx.pow(&1, 2), Nx.pow(&1, 3))))

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
        fn x ->
          x
          |> Nx.dot(w1)
          |> Nx.add(b1)
          |> Nx.dot(w2)
          |> Nx.add(b2)
          |> Nx.multiply(labels)
          |> Nx.sum()
        end
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
        fn b1 ->
          b1
          |> Nx.dot(w2)
          |> Nx.multiply(labels)
          |> Nx.sum()
        end
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

    defn grad_batched_dot_rule_lhs(t1, t2) do
      grad(t1, &Nx.sum(Nx.dot(&1, [1], [0], t2, [2], [0])))
    end

    test "computes the gradient with dot with batching" do
      assert grad_batched_dot_rule_lhs(
               Nx.iota({3, 2, 4}, type: {:f, 32}),
               Nx.iota({3, 3, 2}, type: {:f, 32})
             ) ==
               Nx.tensor([
                 [[6.0, 6.0, 6.0, 6.0], [9.0, 9.0, 9.0, 9.0]],
                 [[24.0, 24.0, 24.0, 24.0], [27.0, 27.0, 27.0, 27.0]],
                 [[42.0, 42.0, 42.0, 42.0], [45.0, 45.0, 45.0, 45.0]]
               ])
    end
  end

  describe "conv rule" do
    defn grad_sum_conv_x(x, y), do: grad(x, &Nx.sum(Nx.conv(&1, y)))
    defn grad_sum_conv_y(x, y), do: grad(y, &Nx.sum(Nx.conv(x, &1)))

    test "computes the gradient of the both sides, no padding, no stride" do
      lhs = Nx.iota({1, 3, 2, 2}, type: {:f, 32})
      rhs = Nx.iota({6, 3, 2, 1}, type: {:f, 32})

      assert grad_sum_conv_x(lhs, rhs) ==
               Nx.tensor([
                 [
                   [[90.0, 90.0], [96.0, 96.0]],
                   [[102.0, 102.0], [108.0, 108.0]],
                   [[114.0, 114.0], [120.0, 120.0]]
                 ]
               ])

      assert grad_sum_conv_y(lhs, rhs) ==
               Nx.tensor([
                 [[[1.0], [5.0]], [[9.0], [13.0]], [[17.0], [21.0]]],
                 [[[1.0], [5.0]], [[9.0], [13.0]], [[17.0], [21.0]]],
                 [[[1.0], [5.0]], [[9.0], [13.0]], [[17.0], [21.0]]],
                 [[[1.0], [5.0]], [[9.0], [13.0]], [[17.0], [21.0]]],
                 [[[1.0], [5.0]], [[9.0], [13.0]], [[17.0], [21.0]]],
                 [[[1.0], [5.0]], [[9.0], [13.0]], [[17.0], [21.0]]]
               ])
    end

    defn grad_sum_conv_x_cos_x_sin_y(x, y), do: grad(x, &Nx.sum(Nx.conv(Nx.cos(&1), Nx.sin(y))))
    defn grad_sum_conv_y_cos_x_sin_y(x, y), do: grad(y, &Nx.sum(Nx.conv(Nx.cos(x), Nx.sin(&1))))

    test "computes the gradient of both sides, no padding, no stride, inner function" do
      x = Nx.iota({1, 3, 2, 2}, type: {:f, 32})
      y = Nx.iota({6, 3, 2, 1}, type: {:f, 32})

      lhs = grad_sum_conv_x_cos_x_sin_y(x, y)

      rhs =
        Nx.tensor([
          [
            [[0.0, 2.9119823], [-1.3931458, -0.21621169]],
            [[3.8719478, 4.906042], [1.1166755, -2.625627]],
            [[0.789102, 0.3287015], [-2.6430445, -4.858301]]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_cos_x_sin_y(x, y)

      rhs =
        Nx.tensor([
          [
            [[1.5403023], [-0.7597403]],
            [[0.15396659], [-1.6969188]],
            [[0.6906596], [-0.23675747]]
          ],
          [
            [[1.4789524], [-1.0600916]],
            [[0.05383231], [-1.5617433]],
            [[0.88658834], [-0.00369389]]
          ],
          [
            [[1.2997901], [-1.2759967]],
            [[-0.05059023], [-1.3021601]],
            [[1.011892], [0.22966394]]
          ],
          [
            [[1.0170873], [-1.3902565]],
            [[-0.15098278], [-0.93884766]],
            [[1.0565889], [0.44472685]]
          ],
          [
            [[0.6533639], [-1.3937694]],
            [[-0.23934811], [-0.5007471]],
            [[1.0171185], [0.62436306]]
          ],
          [
            [[0.23759387], [-1.2862552]],
            [[-0.30864716], [-0.02275731]],
            [[0.89662504], [0.7542629]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_conv_x_same_padding(x, y), do: grad(x, &Nx.sum(Nx.conv(&1, y, padding: :same)))
    defn grad_sum_conv_y_same_padding(x, y), do: grad(y, &Nx.sum(Nx.conv(x, &1, padding: :same)))

    test "computes the gradient of both sides, padding, no stride" do
      x = Nx.iota({2, 1, 5, 5}, type: {:f, 32})
      y = Nx.iota({8, 1, 2, 2}, type: {:f, 32})

      lhs = grad_sum_conv_x_same_padding(x, y)

      rhs =
        Nx.tensor([
          [
            [
              [112.0, 232.0, 232.0, 232.0, 232.0],
              [240.0, 496.0, 496.0, 496.0, 496.0],
              [240.0, 496.0, 496.0, 496.0, 496.0],
              [240.0, 496.0, 496.0, 496.0, 496.0],
              [240.0, 496.0, 496.0, 496.0, 496.0]
            ]
          ],
          [
            [
              [112.0, 232.0, 232.0, 232.0, 232.0],
              [240.0, 496.0, 496.0, 496.0, 496.0],
              [240.0, 496.0, 496.0, 496.0, 496.0],
              [240.0, 496.0, 496.0, 496.0, 496.0],
              [240.0, 496.0, 496.0, 496.0, 496.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_same_padding(x, y)

      rhs =
        Nx.tensor([
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]],
          [[[1225.0, 1000.0], [1080.0, 880.0]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_conv_x_general_stride_lhs_dilated(x, y) do
      grad(
        x,
        &Nx.sum(
          Nx.conv(&1, y, strides: [1, 2], padding: [{1, 2}, {-1, 0}], input_dilation: [2, 1])
        )
      )
    end

    defn grad_sum_conv_y_general_stride_lhs_dilated(x, y) do
      grad(
        y,
        &Nx.sum(
          Nx.conv(x, &1, strides: [1, 2], padding: [{1, 2}, {-1, 0}], input_dilation: [2, 1])
        )
      )
    end

    test "computes the gradient of both sides, general padding, stride, lhs dilated" do
      x = Nx.iota({1, 1, 8, 5}, type: {:f, 32})
      y = Nx.iota({2, 1, 2, 2}, type: {:f, 32})

      lhs = grad_sum_conv_x_general_stride_lhs_dilated(x, y)

      rhs =
        Nx.tensor([
          [
            [
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0],
              [0.0, 12.0, 16.0, 12.0, 16.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_general_stride_lhs_dilated(x, y)
      rhs = Nx.tensor([[[[312.0, 328.0], [312.0, 328.0]]], [[[312.0, 328.0], [312.0, 328.0]]]])

      assert_all_close(lhs, rhs)
    end

    defn grad_mean_conv_x_general_stride_rhs_dilated(x, y) do
      grad(
        x,
        &Nx.mean(
          Nx.conv(&1, y, strides: [1, 2], padding: [{1, 2}, {-1, 0}], kernel_dilation: [1, 2])
        )
      )
    end

    defn grad_mean_conv_y_general_stride_rhs_dilated(x, y) do
      grad(
        y,
        &Nx.mean(
          Nx.conv(x, &1, strides: [1, 2], padding: [{1, 2}, {-1, 0}], kernel_dilation: [1, 2])
        )
      )
    end

    test "computes the gradient of both sides, general padding, stride, rhs dilated" do
      x = Nx.iota({1, 1, 8, 5}, type: {:f, 32})
      y = Nx.iota({2, 1, 2, 2}, type: {:f, 32})

      lhs = grad_mean_conv_x_general_stride_rhs_dilated(x, y)

      rhs =
        Nx.tensor([
          [
            [
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0],
              [0.0, 0.6, 0.0, 0.8, 0.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_mean_conv_y_general_stride_rhs_dilated(x, y)

      rhs =
        Nx.tensor([
          [[[7.4000006, 8.2], [7.4000006, 8.2]]],
          [[[7.4000006, 8.2], [7.4000006, 8.2]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_conv_x_channels_last(x, y) do
      grad(
        x,
        &Nx.sum(Nx.conv(&1, y, input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2]))
      )
    end

    defn grad_sum_conv_y_channels_last(x, y) do
      grad(
        y,
        &Nx.sum(Nx.conv(x, &1, input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2]))
      )
    end

    test "computes the gradient of both sides, valid padding, channels last" do
      x = Nx.iota({3, 4, 4, 2}, type: {:f, 32})
      y = Nx.iota({4, 2, 2, 2}, type: {:f, 32})

      lhs = grad_sum_conv_x_channels_last(x, y)

      rhs =
        Nx.tensor([
          [
            [[48.0, 64.0], [100.0, 132.0], [100.0, 132.0], [52.0, 68.0]],
            [[104.0, 136.0], [216.0, 280.0], [216.0, 280.0], [112.0, 144.0]],
            [[104.0, 136.0], [216.0, 280.0], [216.0, 280.0], [112.0, 144.0]],
            [[56.0, 72.0], [116.0, 148.0], [116.0, 148.0], [60.0, 76.0]]
          ],
          [
            [[48.0, 64.0], [100.0, 132.0], [100.0, 132.0], [52.0, 68.0]],
            [[104.0, 136.0], [216.0, 280.0], [216.0, 280.0], [112.0, 144.0]],
            [[104.0, 136.0], [216.0, 280.0], [216.0, 280.0], [112.0, 144.0]],
            [[56.0, 72.0], [116.0, 148.0], [116.0, 148.0], [60.0, 76.0]]
          ],
          [
            [[48.0, 64.0], [100.0, 132.0], [100.0, 132.0], [52.0, 68.0]],
            [[104.0, 136.0], [216.0, 280.0], [216.0, 280.0], [112.0, 144.0]],
            [[104.0, 136.0], [216.0, 280.0], [216.0, 280.0], [112.0, 144.0]],
            [[56.0, 72.0], [116.0, 148.0], [116.0, 148.0], [60.0, 76.0]]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_channels_last(x, y)

      rhs =
        Nx.tensor([
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]],
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]],
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]],
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_conv_x_feature_groups(x, y),
      do: grad(x, &Nx.sum(Nx.conv(&1, y, feature_group_size: 2)))

    defn grad_sum_conv_y_feature_groups(x, y),
      do: grad(y, &Nx.sum(Nx.conv(x, &1, feature_group_size: 2)))

    test "computes the gradient for both sides, feature grouped" do
      x = Nx.iota({1, 4, 4, 4})
      y = Nx.iota({2, 2, 2, 2})

      lhs = grad_sum_conv_x_feature_groups(x, y)

      rhs =
        Nx.tensor([
          [
            [
              [0.0, 1.0, 1.0, 1.0],
              [2.0, 6.0, 6.0, 4.0],
              [2.0, 6.0, 6.0, 4.0],
              [2.0, 5.0, 5.0, 3.0]
            ],
            [
              [4.0, 9.0, 9.0, 5.0],
              [10.0, 22.0, 22.0, 12.0],
              [10.0, 22.0, 22.0, 12.0],
              [6.0, 13.0, 13.0, 7.0]
            ],
            [
              [8.0, 17.0, 17.0, 9.0],
              [18.0, 38.0, 38.0, 20.0],
              [18.0, 38.0, 38.0, 20.0],
              [10.0, 21.0, 21.0, 11.0]
            ],
            [
              [12.0, 25.0, 25.0, 13.0],
              [26.0, 54.0, 54.0, 28.0],
              [26.0, 54.0, 54.0, 28.0],
              [14.0, 29.0, 29.0, 15.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_feature_groups(x, y)

      rhs =
        Nx.tensor([
          [[[45.0, 54.0], [81.0, 90.0]], [[189.0, 198.0], [225.0, 234.0]]],
          [[[333.0, 342.0], [369.0, 378.0]], [[477.0, 486.0], [513.0, 522.0]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_conv_x_batch_groups(x, y) do
      grad(
        x,
        &Nx.sum(Nx.conv(&1, y, batch_group_size: 3, padding: :same, kernel_dilation: [2, 1]))
      )
    end

    defn grad_sum_conv_y_batch_groups(x, y) do
      grad(
        y,
        &Nx.sum(Nx.conv(x, &1, batch_group_size: 3, padding: :same, kernel_dilation: [2, 1]))
      )
    end

    test "computes the gradient for both sides, batch grouped" do
      x = Nx.iota({6, 1, 3, 3})
      y = Nx.iota({3, 1, 2, 2})

      lhs = grad_sum_conv_x_batch_groups(x, y)

      rhs =
        Nx.tensor([
          [[[0.0, 1.0, 1.0], [2.0, 6.0, 6.0], [2.0, 5.0, 5.0]]],
          [[[0.0, 1.0, 1.0], [2.0, 6.0, 6.0], [2.0, 5.0, 5.0]]],
          [[[4.0, 9.0, 9.0], [10.0, 22.0, 22.0], [6.0, 13.0, 13.0]]],
          [[[4.0, 9.0, 9.0], [10.0, 22.0, 22.0], [6.0, 13.0, 13.0]]],
          [[[8.0, 17.0, 17.0], [18.0, 38.0, 38.0], [10.0, 21.0, 21.0]]],
          [[[8.0, 17.0, 17.0], [18.0, 38.0, 38.0], [10.0, 21.0, 21.0]]]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_batch_groups(x, y)

      rhs =
        Nx.tensor([
          [[[84.0, 60.0], [120.0, 84.0]]],
          [[[300.0, 204.0], [336.0, 228.0]]],
          [[[516.0, 348.0], [552.0, 372.0]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_conv_x_arbitrary_permutation(x, y) do
      grad(
        x,
        &Nx.sum(
          Nx.conv(&1, y,
            input_permutation: [1, 2, 0, 3],
            kernel_permutation: [2, 0, 1, 3],
            output_permutation: [3, 0, 2, 1]
          )
        )
      )
    end

    defn grad_sum_conv_y_arbitrary_permutation(x, y) do
      grad(
        y,
        &Nx.sum(
          Nx.conv(x, &1,
            input_permutation: [1, 2, 0, 3],
            kernel_permutation: [2, 0, 1, 3],
            output_permutation: [3, 0, 2, 1]
          )
        )
      )
    end

    test "computes the gradient of both sides with arbitrary permutations" do
      x = Nx.iota({6, 3, 3, 6})
      y = Nx.iota({3, 2, 6, 3})

      lhs = grad_sum_conv_x_arbitrary_permutation(x, y)

      rhs =
        Nx.tensor([
          [
            [
              [45.0, 96.0, 153.0, 153.0, 108.0, 57.0],
              [261.0, 528.0, 801.0, 801.0, 540.0, 273.0],
              [477.0, 960.0, 1449.0, 1449.0, 972.0, 489.0]
            ],
            [
              [45.0, 96.0, 153.0, 153.0, 108.0, 57.0],
              [261.0, 528.0, 801.0, 801.0, 540.0, 273.0],
              [477.0, 960.0, 1449.0, 1449.0, 972.0, 489.0]
            ],
            [
              [45.0, 96.0, 153.0, 153.0, 108.0, 57.0],
              [261.0, 528.0, 801.0, 801.0, 540.0, 273.0],
              [477.0, 960.0, 1449.0, 1449.0, 972.0, 489.0]
            ]
          ],
          [
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ]
          ],
          [
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ]
          ],
          [
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ]
          ],
          [
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ],
            [
              [198.0, 408.0, 630.0, 630.0, 432.0, 222.0],
              [630.0, 1272.0, 1926.0, 1926.0, 1296.0, 654.0],
              [1062.0, 2136.0, 3222.0, 3222.0, 2160.0, 1086.0]
            ]
          ],
          [
            [
              [153.0, 312.0, 477.0, 477.0, 324.0, 165.0],
              [369.0, 744.0, 1125.0, 1125.0, 756.0, 381.0],
              [585.0, 1176.0, 1773.0, 1773.0, 1188.0, 597.0]
            ],
            [
              [153.0, 312.0, 477.0, 477.0, 324.0, 165.0],
              [369.0, 744.0, 1125.0, 1125.0, 756.0, 381.0],
              [585.0, 1176.0, 1773.0, 1773.0, 1188.0, 597.0]
            ],
            [
              [153.0, 312.0, 477.0, 477.0, 324.0, 165.0],
              [369.0, 744.0, 1125.0, 1125.0, 756.0, 381.0],
              [585.0, 1176.0, 1773.0, 1773.0, 1188.0, 597.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_conv_y_arbitrary_permutation(x, y)

      rhs =
        Nx.tensor([
          [
            [
              [7650.0, 7710.0, 7770.0],
              [7650.0, 7710.0, 7770.0],
              [7650.0, 7710.0, 7770.0],
              [7650.0, 7710.0, 7770.0],
              [7650.0, 7710.0, 7770.0],
              [7650.0, 7710.0, 7770.0]
            ],
            [
              [10890.0, 10950.0, 11010.0],
              [10890.0, 10950.0, 11010.0],
              [10890.0, 10950.0, 11010.0],
              [10890.0, 10950.0, 11010.0],
              [10890.0, 10950.0, 11010.0],
              [10890.0, 10950.0, 11010.0]
            ]
          ],
          [
            [
              [8010.0, 8070.0, 8130.0],
              [8010.0, 8070.0, 8130.0],
              [8010.0, 8070.0, 8130.0],
              [8010.0, 8070.0, 8130.0],
              [8010.0, 8070.0, 8130.0],
              [8010.0, 8070.0, 8130.0]
            ],
            [
              [11250.0, 11310.0, 11370.0],
              [11250.0, 11310.0, 11370.0],
              [11250.0, 11310.0, 11370.0],
              [11250.0, 11310.0, 11370.0],
              [11250.0, 11310.0, 11370.0],
              [11250.0, 11310.0, 11370.0]
            ]
          ],
          [
            [
              [8370.0, 8430.0, 8490.0],
              [8370.0, 8430.0, 8490.0],
              [8370.0, 8430.0, 8490.0],
              [8370.0, 8430.0, 8490.0],
              [8370.0, 8430.0, 8490.0],
              [8370.0, 8430.0, 8490.0]
            ],
            [
              [11610.0, 11670.0, 11730.0],
              [11610.0, 11670.0, 11730.0],
              [11610.0, 11670.0, 11730.0],
              [11610.0, 11670.0, 11730.0],
              [11610.0, 11670.0, 11730.0],
              [11610.0, 11670.0, 11730.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    test "works with complex numbers" do
      t = Nx.reshape(~VEC[1+1i 2 3-3i], {1, 1, 3})
      k = Nx.reshape(~VEC[1 2i 3i], {1, 1, 3})

      assert_all_close(
        Nx.reshape(~VEC[1 2i 3i], {1, 1, 3}),
        grad_sum_conv_x(t, k)
      )

      assert_all_close(
        Nx.reshape(~VEC[1+1i 2 3-3i], {1, 1, 3}),
        grad_sum_conv_y(t, k)
      )

      assert_all_close(
        Nx.reshape(~VEC[-1.0926-0.5343i -3.2978i 99.3534-14.2328i], {1, 1, 3}),
        grad_sum_conv_x_cos_x_sin_y(t, k)
      )

      assert_all_close(
        Nx.reshape(~VEC[0.45046-0.5343i -1.56562 -100.3435+14.2328i], {1, 1, 3}),
        grad_sum_conv_y_cos_x_sin_y(t, k)
      )
    end
  end

  describe "window sum rule" do
    defn grad_sum_window_sum(t), do: grad(t, &Nx.sum(Nx.window_sum(&1, {2, 1, 2, 1})))

    test "computes the gradient of a basic window sum" do
      x = Nx.iota({2, 1, 3, 3}, type: {:f, 32})

      lhs = grad_sum_window_sum(x)

      rhs =
        Nx.tensor([
          [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]],
          [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_sum_cos(t), do: grad(t, &Nx.sum(Nx.window_sum(Nx.cos(&1), {2, 1, 2, 1})))

    test "computes the gradient with an inner function" do
      x = Nx.iota({2, 1, 3, 3}, type: {:f, 32})

      lhs = grad_sum_window_sum_cos(x)

      rhs =
        Nx.tensor([
          [
            [
              [0.0, -0.84147096, -0.9092974],
              [-0.28224, 1.513605, 1.9178486],
              [0.2794155, -0.6569866, -0.98935825]
            ]
          ],
          [
            [
              [-0.4121185, 0.5440211, 0.9999902],
              [1.0731459, -0.84033406, -1.9812148],
              [-0.65028787, 0.2879033, 0.96139747]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_sum_dilated(x),
      do: grad(x, &Nx.sum(Nx.window_sum(&1, {1, 1, 3, 2}, window_dilations: [2, 2, 1, 1])))

    test "computes the gradient with dilations" do
      x = Nx.iota({4, 2, 4, 2}, type: {:f, 32})

      lhs = grad_sum_window_sum_dilated(x)

      rhs =
        Nx.tensor([
          [
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]]
          ],
          [
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]]
          ],
          [
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]]
          ],
          [
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_sum_padding(x) do
      grad(x, &Nx.sum(Nx.window_sum(&1, {2, 1, 1, 2}, padding: [{0, 1}, {2, 1}, {3, 0}, {0, 0}])))
    end

    test "computes the gradien with general padding" do
      x = Nx.iota({3, 2, 1, 2})

      lhs = grad_sum_window_sum_padding(x)

      rhs =
        Nx.tensor([
          [[[1.0, 1.0]], [[1.0, 1.0]]],
          [[[2.0, 2.0]], [[2.0, 2.0]]],
          [[[2.0, 2.0]], [[2.0, 2.0]]]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_sum_stride_padding_dilated(x) do
      grad(
        x,
        &Nx.sum(
          Nx.window_sum(Nx.cos(Nx.sin(&1)), {2, 1, 1, 2},
            strides: [1, 2, 2, 1],
            padding: [{0, 1}, {2, 1}, {3, 0}, {0, 0}],
            window_dilations: [1, 2, 1, 1]
          )
        )
      )
    end

    test "computes the gradient with stride, padding, dilation, inner function" do
      x = Nx.iota({3, 2, 3, 2})

      lhs = grad_sum_window_sum_stride_padding_dilated(x)

      rhs =
        Nx.tensor([
          [
            [[0.0, 0.0], [0.32836998, 0.1392445], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
          ],
          [
            [[0.0, 0.0], [-0.22872283, 0.9198537], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
          ],
          [
            [[0.0, 0.0], [-0.89374965, 0.47741777], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_nested_window_sum(x) do
      grad(x, &Nx.sum(Nx.window_sum(Nx.window_sum(&1, {2, 1, 1, 1}), {1, 2, 1, 1})))
    end

    test "works with nested window sums" do
      x = Nx.iota({4, 3, 4, 4})

      lhs = grad_nested_window_sum(x)

      rhs =
        Nx.tensor([
          [
            [
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0]
            ],
            [
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0]
            ],
            [
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0]
            ]
          ],
          [
            [
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0]
            ],
            [
              [4.0, 4.0, 4.0, 4.0],
              [4.0, 4.0, 4.0, 4.0],
              [4.0, 4.0, 4.0, 4.0],
              [4.0, 4.0, 4.0, 4.0]
            ],
            [
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0]
            ]
          ],
          [
            [
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0]
            ],
            [
              [4.0, 4.0, 4.0, 4.0],
              [4.0, 4.0, 4.0, 4.0],
              [4.0, 4.0, 4.0, 4.0],
              [4.0, 4.0, 4.0, 4.0]
            ],
            [
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0]
            ]
          ],
          [
            [
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0]
            ],
            [
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0]
            ],
            [
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0]
            ]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    test "works with complex" do
      x =
        {2, 1, 3, 3}
        |> Nx.iota(type: {:c, 64})
        |> Nx.multiply(Nx.Constants.i())

      lhs = grad_sum_window_sum(x)

      rhs =
        Nx.tensor([
          [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]],
          [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]]
        ])

      assert_all_close(lhs, rhs)
    end
  end

  describe "window_min/max rule" do
    defn grad_sum_window_max(t), do: grad(t, &Nx.sum(Nx.window_max(&1, {1, 2, 1, 2})))

    test "works with window max, no padding no stride" do
      x = Nx.iota({2, 3, 3, 2}, type: {:f, 32})
      lhs = grad_sum_window_max(x)

      rhs =
        Nx.tensor([
          [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
          ],
          [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_min(t), do: grad(t, &Nx.sum(Nx.window_min(&1, {1, 2, 1, 2})))

    test "works with window min, no padding no stride" do
      x = Nx.iota({2, 3, 3, 2}, type: {:f, 32})
      lhs = grad_sum_window_min(x)

      rhs =
        Nx.tensor([
          [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
          ],
          [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_max_cos(t) do
      grad(
        t,
        &Nx.sum(Nx.window_max(Nx.cos(&1), {1, 1, 2, 2}, padding: :same, strides: [1, 1, 2, 2]))
      )
    end

    test "works with window max, inner function, same padding, stride" do
      x = Nx.iota({1, 4, 4, 2}, type: {:f, 32})
      lhs = grad_sum_window_max_cos(x)

      rhs =
        Nx.tensor([
          [
            [[-0.0, -0.0], [-0.0, -0.0], [0.0, 0.0], [0.2794155, -0.0]],
            [[-0.0, -0.0], [0.0, 0.9999902], [0.0, -0.42016703], [-0.0, -0.0]],
            [[0.0, 0.0], [0.0, -0.1498772], [-0.9129453, -0.0], [0.0, 0.0]],
            [[0.0, 0.13235176], [-0.0, -0.0], [-0.0, 0.0], [0.0, 0.40403765]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_window_min_cos(t) do
      grad(
        t,
        &Nx.sum(Nx.window_min(Nx.cos(&1), {1, 1, 2, 2}, padding: :same, strides: [1, 1, 2, 2]))
      )
    end

    test "works with window min, inner function, same padding, stride" do
      x = Nx.iota({1, 4, 4, 2}, type: {:f, 32})
      lhs = grad_sum_window_min_cos(x)

      rhs =
        Nx.tensor([
          [
            [[-0.0, -0.0], [-0.0, -0.14112000167369843], [0.756802499294281, 0.0], [0.0, -0.0]],
            [[-0.0, -0.41211849451065063], [0.0, 0.0], [0.0, -0.0], [-0.0, -0.6502878665924072]],
            [[0.2879033088684082, 0.0], [0.0, -0.0], [-0.0, -0.0], [0.008851309306919575, 0.0]],
            [[0.0, 0.0], [-0.0, -0.9563759565353394], [-0.2709057927131653, 0.0], [0.0, 0.0]]
          ]
        ])

      assert_all_close(lhs, rhs)
    end
  end

  describe "chain rule" do
    defn grad_tanh_exp(t), do: grad(t, &Nx.tanh(Nx.exp(&1)))

    test "computes gradient" do
      assert grad_tanh_exp(Nx.tensor(1.0)) == Nx.tensor(0.046936701983213425)

      for _ <- @iters do
        t = random_uniform(0.0, 10.0, type: {:f, 64})
        check_grads!(&Nx.tanh(Nx.exp(&1)), &grad_tanh_exp/1, t)
      end
    end
  end

  describe "grad grad" do
    defn grad_tanh_base(t), do: grad(t, &Nx.tanh(&1))
    defn grad_grad_tanh(t), do: grad(t, &grad(&1, fn t -> Nx.tanh(t) end))

    test "computes gradient" do
      assert grad_grad_tanh(Nx.tensor(1.0)) == Nx.tensor(-0.6397000084492246)

      for _ <- @iters do
        t = random_uniform(0.0, 10.0, type: {:f, 64})
        check_grads!(&grad_tanh_base/1, &grad_grad_tanh/1, t)
      end
    end
  end

  describe "tuples" do
    defnp tuple_pattern({a, b}), do: Nx.pow(a, 2) + b
    defn grad_tuple_pattern(t), do: grad(t, &tuple_pattern({&1, 2.0}))

    test "as patterns" do
      assert grad_tuple_pattern(Nx.tensor(1.0)) == Nx.tensor(2.0)
    end

    defn grad_tuple_input(a, b) do
      grad({a, b}, fn {a, b} -> Nx.pow(a, 2) * Nx.pow(b, 3) end)
    end

    defn grad_tuple_input(a, b, c) do
      grad({a, b, c}, fn {a, b, c} -> Nx.pow(a, 2) * Nx.pow(b, 3) * Nx.pow(c, 4) end)
    end

    defn grad_tuple_unused(a, b) do
      grad({a, b}, fn {a, _b} -> Nx.pow(a, a) end)
    end

    test "as multiple inputs" do
      assert grad_tuple_input(Nx.tensor(1.0), Nx.tensor(1.0)) ==
               {Nx.tensor(2.0), Nx.tensor(3.0)}

      assert grad_tuple_input(Nx.tensor(1.0), Nx.tensor(1.0), Nx.tensor(1.0)) ==
               {Nx.tensor(2.0), Nx.tensor(3.0), Nx.tensor(4.0)}

      assert grad_tuple_unused(Nx.tensor(1.0), Nx.tensor(1.0)) ==
               {Nx.tensor(1.0), Nx.tensor(0.0)}
    end

    defn grad_tuple_output(a), do: grad(a, &{&1 + 1, &1 - 1})

    test "raises on tuple output" do
      assert_raise ArgumentError, ~r"expected a tensor or a number", fn ->
        grad_tuple_output(Nx.tensor(1.0))
      end
    end
  end

  for fun <-
        [:cbrt, :cos, :exp, :expm1, :log, :log1p, :sigmoid] ++
          [:mean, :negate, :rsqrt, :sin, :sqrt, :sum, :tanh] do
    describe "#{fun}" do
      grad_fun = :"grad_#{fun}"
      defn unquote(grad_fun)(t), do: grad(t, &(Nx.unquote(fun) / 1))

      test "computes gradient" do
        for _ <- @iters, type <- @types do
          t = random_uniform(0.1, 10.0, type: type)
          check_grads!(&Nx.unquote(fun)(&1), &(__MODULE__.unquote(grad_fun) / 1), t)
        end
      end
    end
  end

  for fun <- [:real, :imag, :conjugate] do
    describe "#{fun}" do
      grad_fun = :"grad_#{fun}"
      defn unquote(grad_fun)(t), do: grad(t, &(Nx.unquote(fun) / 1))

      test "computes gradient" do
        for _ <- @iters do
          t = random_uniform(0.1, 10.0, type: {:c, 128})
          check_grads!(&Nx.unquote(fun)(&1), &(__MODULE__.unquote(grad_fun) / 1), t)
        end
      end
    end
  end

  describe "tan" do
    defn grad_tan(t), do: grad(t, &Nx.tan/1)

    test "computes gradient" do
      for _ <- @iters, type <- @types do
        # check_grads!/4 fails for values close to the asymptotes
        # of tan's gradient, so we select t to avoid them.
        multiplier = random_uniform(0, 10, type: {:u, 32})
        offset = random_uniform(-1.5, 1.5, type: type)
        t = 3.14159 |> Nx.multiply(multiplier) |> Nx.add(offset)
        check_grads!(&Nx.tan/1, &grad_tan/1, t)
      end
    end
  end

  describe "inverse trig family" do
    defn grad_asin(t), do: grad(t, &Nx.asin/1)
    defn grad_acos(t), do: grad(t, &Nx.acos/1)
    defn grad_atan(t), do: grad(t, &Nx.atan/1)

    test "computes gradient of inverse trig functions" do
      for _ <- @iters, type <- @types do
        t = random_uniform(-0.999, 0.999, type: type)
        check_grads!(&Nx.asin/1, &grad_asin/1, t, atol: 1.0e-5, rtol: 1.0e-2)
        check_grads!(&Nx.atan/1, &grad_atan/1, t, atol: 0.1, rtol: 1.0e-2)
        check_grads!(&Nx.atan/1, &grad_atan/1, Nx.multiply(1000.0, t), atol: 1.0e-2)
      end
    end

    test "computes gradient for acos" do
      # acos/1 needs to be tested separately because
      # check_grads! yields the wrong result, even though
      # the formula is in accordance to references.

      for _ <- @iters, type <- @types do
        t = random_uniform(-0.999, 0.999, type: type)

        expected = t |> Nx.pow(2) |> Nx.negate() |> Nx.add(1) |> Nx.rsqrt() |> Nx.negate()

        assert_all_close(grad_acos(t), expected)
      end
    end
  end

  describe "hyperbolics" do
    defn grad_sinh(t), do: grad(t, &Nx.sinh/1)
    defn grad_cosh(t), do: grad(t, &Nx.cosh/1)

    test "computes gradient" do
      for _ <- @iters, type <- @types do
        t = random_uniform(-10, 10, type: type)
        check_grads!(&Nx.sinh/1, &grad_sinh/1, t)
        check_grads!(&Nx.cosh/1, &grad_cosh/1, t)
      end
    end
  end

  describe "inverse hyperbolic functions" do
    defn grad_asinh(t), do: grad(t, &Nx.asinh/1)
    defn grad_acosh(t), do: grad(t, &Nx.acosh/1)
    defn grad_atanh(t), do: grad(t, &Nx.atanh/1)

    test "computes gradient of inverse hyperbolic functions" do
      for _ <- @iters, type <- @types do
        t = random_uniform(-100.0, 100.0, type: type)
        check_grads!(&Nx.asinh/1, &grad_asinh/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end

      for _ <- @iters, type <- @types do
        t = random_uniform(1.01, 100.0, type: type)
        check_grads!(&Nx.acosh/1, &grad_acosh/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end

      for _ <- @iters, type <- @types do
        t = random_uniform(-0.999, 0.999, type: type)
        check_grads!(&Nx.atanh/1, &grad_atanh/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end
  end

  describe "erf" do
    defn grad_erf(t), do: grad(t, &Nx.erf/1)

    test "computes the gradient" do
      for _ <- @iters do
        t = random_uniform(-100.0, 100.0, type: {:f, 64})
        check_grads!(&Nx.erf/1, &grad_erf/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end
  end

  describe "erfc" do
    defn grad_erfc(t), do: grad(t, &Nx.erfc/1)

    test "computes the gradient" do
      for _ <- @iters do
        t = random_uniform(-100.0, 100.0, type: {:f, 64})
        check_grads!(&Nx.erfc/1, &grad_erfc/1, t, atol: 1.0e-4)
      end
    end
  end

  describe "erf_inv" do
    defn grad_erf_inv(t), do: grad(t, &Nx.erf_inv/1)

    test "computes gradient close to 0.0" do
      for _ <- @iters do
        t = random_uniform(0.0, 0.9, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end

    test "computes gradient between 0.9 and 0.95" do
      for _ <- @iters do
        t = random_uniform(0.9, 0.95, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end

    test "computes gradient between 0.95 and 0.98" do
      for _ <- @iters do
        t = random_uniform(0.95, 0.98, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end

    test "computes gradient approaching 1.0 but is sharply curved" do
      # check_grads! does not work near 1 due to sharp curve between close x's
      coords = [
        {0.9, 3.43},
        {0.98, 13.26},
        {0.99, 24.45},
        {0.991, 26.85},
        {0.992, 29.84},
        {0.993, 33.64},
        {0.994, 38.64},
        {0.995, 45.56},
        {0.999, 198.94}
      ]

      for {x, y} <- coords do
        assert_in_delta(Nx.to_number(grad_erf_inv(x)), y, 0.01)
      end
    end
  end

  describe "broadcast" do
    defn grad_sum_broadcast(t), do: grad(t, &Nx.sum(Nx.broadcast(&1, {3, 2, 2})))

    test "computes gradient" do
      for multiplier <- [1, Complex.new(0, 1)] do
        assert grad_sum_broadcast({3, 2, 2} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(1.0, {3, 2, 2})

        assert grad_sum_broadcast({1, 2, 2} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(3.0, {1, 2, 2})

        assert grad_sum_broadcast({3, 1, 2} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(2.0, {3, 1, 2})

        assert grad_sum_broadcast({3, 2, 1} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(2.0, {3, 2, 1})

        assert grad_sum_broadcast({3, 1, 1} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(4.0, {3, 1, 1})

        assert grad_sum_broadcast({1, 1, 1} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(12.0, {1, 1, 1})

        assert grad_sum_broadcast({2, 2} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(3.0, {2, 2})

        assert grad_sum_broadcast({1, 2} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(6.0, {1, 2})

        assert grad_sum_broadcast({2, 1} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(6.0, {2, 1})

        assert grad_sum_broadcast({2} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(6.0, {2})

        assert grad_sum_broadcast({} |> Nx.iota() |> Nx.multiply(multiplier)) ==
                 Nx.broadcast(12.0, {})
      end
    end
  end

  describe "concatenate" do
    defn concatenate_grad(t) do
      grad(t, &Nx.sum(Nx.concatenate([&1, &1], axis: 0)))
    end

    defn concatenate_grad_power(t) do
      grad(
        t,
        &Nx.sum(Nx.concatenate([Nx.pow(&1, 2), Nx.pow(&1, 3)], axis: 0))
      )
    end

    defn concatenate_grad_composed(t) do
      grad(
        t,
        &Nx.sum(Nx.concatenate([Nx.log(Nx.pow(&1, 2)), Nx.log(Nx.pow(&1, 3))], axis: 3))
      )
    end

    defn concatenate_grad_log1p(t) do
      grad(
        t,
        &Nx.sum(Nx.log1p(Nx.concatenate([Nx.pow(&1, 2), Nx.pow(&1, 3)], axis: 3)))
      )
    end

    test "computes grad for {1}-tensor" do
      assert concatenate_grad(Nx.tensor([1.0])) == Nx.tensor([2.0])
    end

    test "computes grad for {2, 2} tensor" do
      assert concatenate_grad(Nx.tensor([[1.0, 2.0], [3.0, 4.0]])) ==
               Nx.tensor([[2.0, 2.0], [2.0, 2.0]])
    end

    test "computes grad for powers of a {2, 2}-tensor" do
      assert concatenate_grad_power(Nx.tensor([[1.0, 2.0], [3.0, 4.0]])) ==
               Nx.tensor([[5.0, 16.0], [33.0, 56.0]])

      assert concatenate_grad_power(~MAT[
        1i 2
        3 4i
      ]) == ~MAT[
        -3+2i 16
        33 -48+8i
      ]
    end

    test "computes grad for composed functions on a multidim tensor" do
      assert concatenate_grad_composed(Nx.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])) ==
               Nx.tensor([[[[5.0, 2.5], [1.6666667, 1.25], [1.0, 0.8333334]]]])
    end

    test "computes grad for composed functions applied to concatenate" do
      assert concatenate_grad_log1p(Nx.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])) ==
               Nx.tensor([[[[2.5, 2.1333334], [1.5642858, 1.2090498], [0.9798535, 0.8220202]]]])
    end
  end

  describe "stack" do
    test "works on compound functions for more than 1 axis" do
      # This is a test that ensures that the added axis from the
      # stack operation is correctly squeezed back out by
      # the gradient computation.
      x = 2.0

      assert grad(Nx.tensor([[x]]), fn t ->
               a = Nx.pow(t, 2)
               b = Nx.pow(t, 3)
               c = Nx.pow(t, 4)

               Nx.stack([a, b, c], axis: 1)
               |> Nx.sum()
             end) == Nx.tensor([[2 * x + 3 * x ** 2 + 4 * x ** 3]])
    end
  end

  describe "cholesky" do
    defn cholesky_grad(t) do
      grad(t, fn x -> x |> Nx.LinAlg.cholesky() |> Nx.sum() end)
    end

    defn cholesky_cos_grad(t) do
      grad(t, fn x -> x |> Nx.LinAlg.cholesky() |> Nx.cos() |> Nx.sum() end)
    end

    defn exp_cholesky_grad(t) do
      grad(t, fn x -> x |> Nx.exp() |> Nx.LinAlg.cholesky() |> Nx.sum() end)
    end

    test "computes for 2x2 complex matrix" do
      t = ~MAT[
        1 -2i
        2i 5
      ]

      assert cholesky_grad(t) == ~MAT[
        2.5-1i -1i
        1+1i 0.5
      ]

      assert_all_close(cholesky_cos_grad(t), ~MAT[
        -5.7305   0.8414i
        -4.4683i -0.4207
      ])
    end

    test "computes grad for {2, 2}-tensor" do
      t = Nx.tensor([[1.0, 2.0], [2.0, 5.0]])
      assert cholesky_grad(t) == Nx.tensor([[1.5, -1.0], [0.0, 0.5]])
    end

    test "computes grad for composed {2,2}-tensor" do
      t = Nx.tensor([[1.0, 2.0], [2.0, 5.0]])

      assert cholesky_cos_grad(t) ==
               Nx.tensor([[-1.1943799, 0.84147096], [-0.06782645, -0.42073548]])

      assert exp_cholesky_grad(t) ==
               Nx.tensor([[-0.5299541, -0.88652998], [3.59515929, 6.550619]])
    end
  end

  describe "qr" do
    defn qr_grad(t) do
      grad(t, fn tensor ->
        {q, r} = Nx.LinAlg.qr(tensor)

        q
        |> Nx.add(r)
        |> Nx.sum()
      end)
    end

    defn qr_megapower_grad(t) do
      grad(t, fn tensor ->
        {q, r} = Nx.LinAlg.qr(Nx.pow(tensor, 2))

        q
        |> Nx.cos()
        |> Nx.add(Nx.sin(r))
        |> Nx.sum()
      end)
    end

    test "computes grad for tensor" do
      assert_all_close(
        qr_grad(Nx.tensor([[1.0, 2.0], [1.0, -1.0]])),
        Nx.tensor([
          [0.70709, 1.4142],
          [0.70709, 0.0]
        ])
      )
    end

    test "computes qr_megapower_grad for tensor" do
      assert_all_close(
        qr_megapower_grad(Nx.tensor([[1.0, 2.0], [1.0, -1.0]])),
        Nx.tensor([
          [0.1112, -4.0914],
          [0.3298, 0.5660]
        ])
      )
    end

    test "computes qr_megapower_grad for complex tensor" do
      assert_all_close(
        qr_megapower_grad(~MAT[
          1 2i
          3 4i
        ]),
        ~MAT[
          1.94476 2.72264i
          4.98145 5.87086i
        ]
      )
    end
  end

  describe "lu" do
    defn lu_grad(t) do
      grad(t, fn tensor ->
        {p, l, u} = Nx.LinAlg.lu(tensor)

        l
        |> Nx.add(u)
        |> Nx.add(p)
        |> Nx.sum()
      end)
    end

    defn lu_megapower_grad(t) do
      grad(t, fn tensor ->
        {p, l, u} = Nx.LinAlg.lu(Nx.pow(tensor, 2))

        l
        |> Nx.cos()
        |> Nx.add(Nx.sin(u))
        |> Nx.subtract(Nx.exp(p))
        |> Nx.sum()
      end)
    end

    test "computes grad for tensor" do
      assert_all_close(
        lu_grad(Nx.tensor([[1.0, 2.0], [1.0, -1.0]])),
        Nx.tensor([
          [2.0, 0.0],
          [-1.0, 1.0]
        ])
      )
    end

    test "computes lu_megapower_grad for tensor" do
      assert_all_close(
        lu_megapower_grad(Nx.tensor([[1.0, 2.0], [1.0, -1.0]])),
        Nx.tensor([
          [-5.1563935, 1.3453956],
          [6.236998, 1.979985]
        ])
      )
    end

    test "computes lu_megapower_grad for complex tensor" do
      assert_all_close(
        lu_megapower_grad(~MAT[
            1i 2
            1 4i
          ]),
        ~MAT[
          6.1484942i 0.76084137
          5.0678897 6.7508316i
        ]
      )
    end
  end

  describe "svd" do
    defn svd_grad(t) do
      grad(t, fn tensor ->
        {u, s, vt} = Nx.LinAlg.svd(tensor)

        s
        |> Nx.make_diagonal()
        |> Nx.add(u)
        |> Nx.add(vt)
        |> Nx.sum()
      end)
    end

    defn svd_composed_grad(t) do
      grad(t, fn tensor ->
        {u, s, vt} = Nx.LinAlg.svd(tensor)

        s
        |> Nx.make_diagonal()
        |> Nx.exp()
        |> Nx.sum()
        |> Nx.add(Nx.cos(u) |> Nx.sum())
        |> Nx.add(Nx.sin(vt) |> Nx.sum())
      end)
    end

    test "computes grad for tensor" do
      assert_all_close(
        svd_grad(Nx.tensor([[3, 0], [1, 2]])),
        Nx.tensor([
          [1.368404507637024, -0.5419228672981262],
          [-0.2197188436985016, 0.6067624092102051]
        ])
      )
    end

    test "computes the composed grad for tensor" do
      assert_all_close(
        svd_composed_grad(Nx.tensor([[3, 0], [1, 2]])),
        Nx.tensor([
          [22.86724090576172, 3.655829906463623],
          [10.035255432128906, 8.769235610961914]
        ])
      )
    end

    test "computes the composed grad for tall tensor" do
      assert_all_close(
        svd_composed_grad(Nx.tensor([[3, 0], [1, 2], [1, 1]])),
        Nx.tensor([
          [25.911056518554688, 6.1099162101745605],
          [12.69705581665039, 10.84456729888916],
          [10.668402671813965, 6.426826477050781]
        ])
      )
    end
  end

  describe "invert" do
    defn invert_grad(t) do
      grad(t, fn tensor ->
        tensor
        |> Nx.LinAlg.invert()
        |> Nx.sum()
      end)
    end

    defn invert_abs_grad(t) do
      grad(t, fn tensor ->
        tensor
        |> Nx.LinAlg.invert()
        |> Nx.abs()
        |> Nx.sum()
      end)
    end

    defn composed_invert_grad(t) do
      grad(t, fn tensor ->
        tensor
        |> Nx.exp()
        |> Nx.sin()
        |> Nx.LinAlg.invert()
        |> Nx.cos()
        |> Nx.log()
        |> Nx.sum()
      end)
    end

    test "computes grad for tensor" do
      assert_all_close(
        invert_grad(Nx.tensor([[1.0, 2.0, 3.0], [0, 4, 5], [0, 0, -6]])),
        Nx.tensor([
          [-0.5833, -0.4583, 0.1667],
          [0.14583, 0.1146, -0.0417],
          [-0.0729, -0.0573, 0.0208]
        ])
      )
    end

    test "computes grad for complex tensor" do
      assert_all_close(
        invert_grad(~MAT[
          1i 1i
          0 1
        ]),
        ~MAT[
          1+i -1i
          0 0
        ]
      )
    end

    test "computes composed function grad for tensor" do
      assert_all_close(
        composed_invert_grad(Nx.tensor([[1.0, 2.0, 3.0], [0, 4, 5], [0, 0, -6]])),
        Nx.tensor([
          [-8.2713, -21.8449, 72.5134],
          [0.9508, 69.7985, -524.3346],
          [-1.2254, 3.1769, -0.0230]
        ])
      )
    end
  end

  describe "squeeze" do
    defn grad_sum_squeeze_broadcast(t),
      do: grad(t, &Nx.sum(Nx.squeeze(Nx.broadcast(&1, {3, 2, 2}))))

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

    test "computes gradient for complex" do
      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({3, 2, 2}))) ==
               Nx.broadcast(1.0, {3, 2, 2})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({1, 2, 2}))) ==
               Nx.broadcast(3.0, {1, 2, 2})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({1, 1, 2}))) ==
               Nx.broadcast(6.0, {1, 1, 2})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({1, 1, 1}))) ==
               Nx.broadcast(12.0, {1, 1, 1})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({2, 2}))) ==
               Nx.broadcast(3.0, {2, 2})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({1, 2}))) ==
               Nx.broadcast(6.0, {1, 2})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({1, 1}))) ==
               Nx.broadcast(12.0, {1, 1})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({2}))) ==
               Nx.broadcast(6.0, {2})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({1}))) ==
               Nx.broadcast(12.0, {1})

      assert grad_sum_squeeze_broadcast(Nx.multiply(Nx.Constants.i(), Nx.iota({}))) ==
               Nx.broadcast(12.0, {})
    end
  end

  describe "pad" do
    defn grad_sum_pad(t), do: grad(t, &Nx.sum(Nx.pad(&1, 2.0, [{-1, 1, 0}, {1, 1, 0}])))
    defn grad_interior_pad(t), do: grad(t, &Nx.sum(Nx.pad(&1, 2.0, [{0, 0, 1}, {0, 0, 1}])))
    defn grad_lots_of_pad(t), do: grad(t, &Nx.sum(Nx.pad(&1, 2.0, [{-2, 1, 4}, {1, 3, 2}])))

    defn grad_pad_fun(t),
      do: grad(t, &Nx.mean(Nx.pad(&1, Nx.mean(Nx.cos(&1)), [{-2, 1, 4}, {1, 3, 2}])))

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
                 [-0.13259540498256683, -0.14328321814537048, -0.022237088531255722],
                 [0.13743554055690765, 0.16928502917289734, 0.062210917472839355]
               ])
    end

    test "works on complex" do
      lhs =
        [[1.0, 2.0], [1.0, 2.0]]
        |> Nx.tensor(type: {:c, 64})
        |> Nx.multiply(Nx.Constants.i())
        |> grad_sum_pad

      rhs = Nx.tensor([[0.0, 0.0], [1.0, 1.0]])

      assert lhs == rhs
    end
  end

  describe "slice" do
    defn grad_mean_slice(t), do: grad(t, &Nx.mean(Nx.slice(&1, [0, 1], [1, 2], strides: [1, 2])))

    defn grad_mean_dynamic_slice(t),
      do: grad(t, &Nx.mean(Nx.slice(&1, [Nx.tensor(0), Nx.tensor(1)], [1, 2], strides: [1, 2])))

    defn grad_sum_slice(t), do: grad(t, &Nx.sum(Nx.slice(&1, [1, 0], [1, 2], strides: [1, 1])))

    defn grad_sum_dynamic_slice(t),
      do: grad(t, &Nx.sum(Nx.slice(&1, [Nx.tensor(1), Nx.tensor(0)], [1, 2], strides: [1, 1])))

    defn grad_sum_pad_slice(t) do
      grad(
        t,
        fn t ->
          Nx.slice(t, [1, 0], [1, 2], strides: [1, 1])
          |> Nx.pad(Nx.mean(Nx.sin(t)), [{2, 1, 2}, {-1, 2, 0}])
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_pad_dynamic_slice(t) do
      grad(
        t,
        fn t ->
          Nx.slice(t, [Nx.tensor(1), Nx.tensor(0)], [1, 2], strides: [1, 1])
          |> Nx.pad(Nx.mean(Nx.sin(t)), [{2, 1, 2}, {-1, 2, 0}])
          |> Nx.sum()
        end
      )
    end

    test "computes gradient" do
      assert grad_mean_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

      assert grad_mean_dynamic_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

      assert grad_sum_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

      assert grad_sum_slice(
               Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
               |> Nx.multiply(Nx.Constants.i())
             ) ==
               Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

      assert grad_sum_dynamic_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

      lhs = grad_sum_pad_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

      rhs =
        Nx.tensor([
          [0.9905542274249228, -0.7629358670030943, -1.8149862437674833],
          [-1.1983466382499552, 1.520047340015915, 1.7603121921923377]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_sum_pad_dynamic_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

      assert_all_close(lhs, rhs)
    end

    defn grad_of_index(t, index), do: grad(index, &Nx.mean(Nx.slice(t, [&1], [2])))

    test "computes gradient of index" do
      assert grad_of_index(
               Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
               Nx.reduce_max(Nx.tensor([-1, 0, 1]))
             ) == Nx.tensor(0.0)
    end
  end

  describe "put_slice" do
    defn grad_mean_put_slice_operand(t1, t2), do: grad(t1, &Nx.mean(Nx.put_slice(&1, [0, 1], t2)))
    defn grad_mean_put_slice_update(t1, t2), do: grad(t2, &Nx.mean(Nx.put_slice(t1, [0, 1], &1)))

    defn grad_sum_pad_put_slice_cos_operand(t1, t2) do
      grad(t1, fn t ->
        t
        |> Nx.cos()
        |> Nx.put_slice([1, 2], Nx.sin(t2))
        |> Nx.pad(Nx.mean(Nx.sin(t)), [{2, 1, 2}, {-1, 2, 0}])
        |> Nx.sum()
      end)
    end

    defn grad_sum_pad_put_slice_sin_update(t1, t2) do
      grad(t2, fn t ->
        t1
        |> Nx.cos()
        |> Nx.put_slice([1, 2], Nx.sin(t))
        |> Nx.pad(Nx.mean(Nx.sin(t1)), [{2, 1, 2}, {-1, 2, 0}])
        |> Nx.sum()
      end)
    end

    test "computes gradient of operand" do
      lhs =
        grad_mean_put_slice_operand(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[10.0, 11.0]])
        )

      rhs = Nx.tensor([[0.16666667, 0.0, 0.0], [0.16666667, 0.16666667, 0.16666667]])

      assert_all_close(lhs, rhs)

      lhs =
        grad_sum_pad_put_slice_cos_operand(
          Nx.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 10.0]]),
          Nx.tensor([[10.0, 11.0], [12.0, 13.0]])
        )

      rhs =
        Nx.tensor([
          [1.84603288, -2.33113245, -3.52359437, -1.47647988],
          [-2.23328237, 1.92810341, 3.28058181, 2.5758327],
          [2.5758327, -1.48648336, -3.11302839, -2.86682772]
        ])

      assert_all_close(lhs, rhs)
    end

    test "computes gradient of update" do
      lhs =
        grad_mean_put_slice_update(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[10.0, 11.0]])
        )

      rhs = Nx.tensor([[0.16666667, 0.16666667]])

      assert_all_close(lhs, rhs)

      lhs =
        grad_sum_pad_put_slice_sin_update(
          Nx.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 10.0]]),
          Nx.tensor([[10.0, 11.0], [12.0, 13.0]])
        )

      rhs = Nx.tensor([[-0.83907153, 0.0044257], [0.84385396, 0.90744678]])

      assert_all_close(lhs, rhs)
    end

    test "works on complex" do
      t1 =
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        |> Nx.tensor(type: {:c, 64})
        |> Nx.multiply(Nx.Constants.i())

      t2 =
        [[10.0, 11.0]]
        |> Nx.tensor(type: {:c, 64})
        |> Nx.multiply(Nx.Constants.i())

      lhs = grad_mean_put_slice_operand(t1, t2)
      rhs = Nx.tensor([[0.16666667, 0.0, 0.0], [0.16666667, 0.16666667, 0.16666667]])

      assert_all_close(lhs, rhs)

      lhs = grad_mean_put_slice_update(t1, t2)
      rhs = Nx.tensor([[0.16666667, 0.16666667]])

      assert_all_close(lhs, rhs)
    end
  end

  describe "reverse" do
    defn grad_sum_reverse_exp(t), do: grad(t, &Nx.sum(Nx.reverse(Nx.exp(&1))))
    defn grad_sum_exp_reverse(t), do: grad(t, &Nx.sum(Nx.exp(Nx.reverse(&1))))

    test "computes gradient" do
      assert grad_sum_exp_reverse(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([
                 [2.718281828459045, 7.38905609893065, 20.085536923187668],
                 [54.598150033144236, 148.4131591025766, 403.4287934927351]
               ])

      assert_all_close(grad_sum_exp_reverse(~VEC[1 -2i 3]), ~VEC[2.7182 -0.4161-0.9092i 20.0855])

      assert grad_sum_reverse_exp(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([
                 [2.718281828459045, 7.38905609893065, 20.085536923187668],
                 [54.598150033144236, 148.4131591025766, 403.4287934927351]
               ])
    end
  end

  describe "abs" do
    defn abs_scalar(t), do: Nx.abs(t)
    defn grad_abs_scalar(t), do: grad(t, &Nx.abs(&1))
    defn grad_abs_squared(t), do: grad(t, &(&1 |> Nx.abs() |> Nx.pow(2)))
    defn grad_abs(t), do: grad(t, &Nx.sum(Nx.abs(&1)))
    defn grad_cos_abs_sin(t), do: grad(t, &Nx.sum(Nx.cos(Nx.abs(Nx.sin(&1)))))

    test "computes gradient with scalars" do
      for _ <- @iters do
        check_grads!(
          &abs_scalar/1,
          &grad_abs_scalar/1,
          random_uniform(0.0, 1000.0, type: {:f, 64})
        )
      end
    end

    test "computes gradient with tensors" do
      assert grad_abs(Nx.tensor([[1.0, 2.0], [3.0, 4.0]])) == Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      assert grad_abs(Nx.tensor([[-1.0, 2.0], [-3.0, 4.0]])) ==
               Nx.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    end

    test "works with complex" do
      assert_all_close(grad_abs(~VEC[0 1i 2]), ~VEC[0 -1i 1])

      # Ensure our definition is in accordance with the definition
      # for abs_squared at the JuliaDiff reference: grad(sum(abs(t))) = 2conj(t)
      assert_all_close(grad_abs_squared(~VEC[0 1 2]), ~VEC[0 2 4])
      assert_all_close(grad_abs_squared(~VEC[0 1i 2 -3i]), ~VEC[0 -2i 4 6i])

      t = ~VEC[0 1+i 2+2i 3+3i]

      assert_all_close(
        grad_abs(t),
        ~VEC[0 0.7071-0.7071i 0.7071-0.7071i 0.7071-0.7071i]
      )

      assert_all_close(
        grad_cos_abs_sin(t),
        ~VEC[0 -0.3120795+1.2447729i -0.05693477-2.053038i  -0.00780554-5.6348686i]
      )
    end
  end

  describe "max" do
    defn grad_max(t), do: grad(t, &Nx.sum(Nx.max(Nx.pow(&1, 2), Nx.pow(&1, 3))))

    test "computes gradient with tensors" do
      assert grad_max(Nx.tensor([[1.0], [2.0], [3.0]])) == Nx.tensor([[2.5], [12.0], [27.0]])

      assert grad_max(Nx.tensor([[1.25, 2.5, 2.75], [1.0, 4.0, 6.0], [2.0, 3.0, 2.0]])) ==
               Nx.tensor([[4.6875, 18.75, 22.6875], [2.5, 48.0, 108.0], [12.0, 27.0, 12.0]])
    end
  end

  describe "min" do
    defn grad_min(t), do: grad(t, &Nx.sum(Nx.min(Nx.pow(&1, 2), Nx.pow(&1, 3))))

    test "computes gradient with tensors" do
      assert grad_min(Nx.tensor([[1.0], [2.0], [3.0]])) == Nx.tensor([[2.5], [4.0], [6.0]])

      assert grad_min(Nx.tensor([[1.25, 2.5, 2.75], [1.0, 4.0, 6.0], [2.0, 3.0, 2.0]])) ==
               Nx.tensor([[2.5, 5.0, 5.5], [2.5, 8.0, 12.0], [4.0, 6.0, 4.0]])
    end
  end

  describe "select rule" do
    defn grad_sum_select(t),
      do: grad(t, &Nx.sum(Nx.select(Nx.greater(&1, 0.0), Nx.exp(&1), Nx.cos(&1))))

    defn grad_sum_select_2by2(t),
      do:
        grad(
          t,
          &Nx.sum(Nx.select(Nx.tensor([[1, 1], [1, 0]], type: {:u, 8}), Nx.exp(&1), Nx.cos(&1)))
        )

    defn grad_max_select(t),
      do: grad(t, &Nx.reduce_max(Nx.select(Nx.greater(&1, 0.0), Nx.exp(&1), Nx.cos(&1))))

    defn grad_sum_select_sum(t),
      do:
        grad(
          t,
          &Nx.sum(Nx.select(Nx.greater(Nx.sum(&1, axes: [0]), 0.0), Nx.sum(&1, axes: [0]), 0.0))
        )

    defn leaky_relu(t), do: Nx.select(t > 0, t, t * 0.1)
    defn grad_nested_select(t), do: grad(t, &Nx.sum(leaky_relu(leaky_relu(&1))))

    # pred is not part of grad
    defn grad_pred_select(t), do: grad(t, &Nx.sum(Nx.select(&1 * &1, 1.0, -1.0)))

    test "computes gradient with sum+select" do
      lhs = grad_sum_select(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]]))

      rhs =
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

      assert_all_close(lhs, rhs)
    end

    test "computes gradient with sum+select for complex tensor" do
      lhs = grad_sum_select_2by2(~MAT[
          1+i 2+i
          3+i -1-4i
        ])

      rhs = ~MAT[
        1.4686+2.2873i 3.9923+6.2176i
        10.8522+16.9013i 22.9790+14.7448i
      ]

      assert_all_close(lhs, rhs)
    end

    test "computes the gradient with max+select" do
      lhs = grad_max_select(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]]))
      rhs = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 148.4131591025766, 0.0]])

      assert_all_close(lhs, rhs)
    end

    test "computes the gradient with sum+select+sum" do
      lhs =
        grad_sum_select_sum(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]]))

      rhs = Nx.tensor([[0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]])

      assert_all_close(lhs, rhs)
    end

    test "computes the gradient with sum+select+select" do
      lhs = grad_nested_select(Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
      rhs = Nx.tensor([0.01, 0.01, 0.01, 1.0, 1.0])
      assert_all_close(lhs, rhs)
    end

    test "is zero when part of pred only" do
      assert grad_pred_select(Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])) ==
               Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    end
  end

  describe "as_type/bitcast" do
    defn grad_as_type(t), do: grad(t, &Nx.sum(Nx.as_type(&1, {:f, 32})))
    defn grad_as_type_complex(t), do: grad(t, &Nx.sum(Nx.as_type(&1, {:c, 64})))
    defn grad_bitcast(t), do: grad(t, &Nx.sum(Nx.bitcast(&1, {:f, 32})))

    defn grad_as_type_downcast(t), do: grad(t, &Nx.sum(Nx.cos(Nx.as_type(&1, {:f, 32}))))

    test "as_type takes the real part when downcasting complex" do
      # Note that, due to the way the grad_as_type_downcast defn is defined,
      # the expected grad is the same as the grad for:
      # Nx.sum(Nx.cos(~VEC[1 2 3])), which is -sin(~VEC[1 2 3])
      t = ~VEC[1+i 2+i 3+i]
      grad = grad_as_type_downcast(t)

      assert grad == grad_as_type_downcast(~VEC[1 2 3])
      assert grad == Nx.negate(Nx.sin(Nx.real(t)))
    end

    test "as_type passes through for non-downcasting calls" do
      assert grad_as_type(Nx.tensor([1, 2, 3])) == Nx.tensor([1.0, 1.0, 1.0])
      assert grad_as_type_complex(~VEC[1+i 2+i 3+i]) == Nx.tensor([1.0, 1.0, 1.0])
    end

    test "bitcast passes through" do
      assert grad_bitcast(Nx.tensor([1, 2, 3], type: {:s, 32})) == Nx.tensor([1.0, 1.0, 1.0])
    end
  end

  describe "if" do
    defn grad_if(t), do: grad(t, fn t -> if(t + 1, do: Nx.pow(t, 2), else: Nx.pow(t, 3)) end)

    defn grad_sum_if(t) do
      grad(t, fn t -> Nx.sum(if(Nx.all(t), do: Nx.pow(t, 2), else: Nx.pow(t, 3))) end)
    end

    defn grad_if_sum(t) do
      grad(t, fn t -> if(Nx.all(t), do: Nx.sum(Nx.pow(t, 2)), else: Nx.sum(Nx.pow(t, 3))) end)
    end

    defn grad_if_tuple(t) do
      grad(t, fn t ->
        {{a, b}, c} =
          if t > 0 do
            {{Nx.pow(t, 2), Nx.pow(t, 3)}, Nx.pow(t, 4)}
          else
            {{Nx.pow(t, 4), Nx.pow(t, 3)}, Nx.pow(t, 2)}
          end

        d = if t > 0, do: 123, else: 456

        a * b + c - d
      end)
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
      assert grad_if_tuple(Nx.tensor(2)) == Nx.tensor(112.0)
      assert grad_if_tuple(Nx.tensor(-1)) == Nx.tensor(5.0)
      assert grad_if_tuple(Nx.tensor(-2)) == Nx.tensor(444.0)
    end
  end

  describe "while" do
    defn grad_while_constant(t) do
      grad(t, fn t ->
        {_, t} =
          while {i = 0, t}, i < 3 do
            {i + 1, t}
          end

        Nx.sum(t)
      end)
    end

    test "computes gradient for constant loop" do
      tensor = Nx.tensor([-2.5, -1.0, 0.0, 1.0, 1.5])
      assert grad_while_constant(tensor) == Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    end

    defn grad_while_param(t, x) do
      value_and_grad(x, fn x ->
        {_, t} =
          while {x, t}, t < 100 do
            {x, x * t}
          end

        Nx.sum(t)
      end)
    end

    test "computes gradient for unrelated param loop" do
      assert grad_while_param(1.0, 2.0) == {Nx.tensor(128.0), Nx.tensor(5461.0)}
    end

    defn grad_while_single_var(t) do
      value_and_grad(t, fn t ->
        while t, t < 100 do
          t * t
        end
      end)
    end

    defn grad_while_single_var_unroll(t) do
      value_and_grad(t, fn v2 ->
        v4 = v2 * v2
        v16 = v4 * v4
        v16 * v16
      end)
    end

    test "computes gradient for single var loop" do
      assert grad_while_single_var(2.0) == grad_while_single_var_unroll(2.0)
    end

    defn grad_while_3x_sin(t) do
      value_and_grad(t, fn t ->
        {_, t} =
          while {i = 0, t = Nx.pow(t, 3)}, i < 3 do
            {i + 1, Nx.sin(t)}
          end

        Nx.sum(Nx.pow(t, 2))
      end)
    end

    defn grad_3x_sin(t) do
      value_and_grad(
        t,
        &(&1 |> Nx.pow(3) |> Nx.sin() |> Nx.sin() |> Nx.sin() |> Nx.pow(2) |> Nx.sum())
      )
    end

    test "computes gradient for while with additional var" do
      # Do not align values so we can find bugs in the number of loops
      tensor = Nx.tensor([-2.5, -1.0, 0.0, 1.0, 1.5])
      {value_while, grad_while} = grad_while_3x_sin(tensor)
      {value_unroll, grad_unroll} = grad_3x_sin(tensor)
      assert value_while == value_unroll
      assert_all_close(grad_while, grad_unroll)
    end
  end

  describe "axes" do
    defn grad_sum_full(t), do: grad(t, &Nx.sum/1)
    defn grad_mean_full(t), do: grad(t, &Nx.mean/1)

    test "computes gradient in full" do
      assert grad_sum_full(Nx.tensor([[1, 2], [3, 4]])) ==
               Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      assert grad_mean_full(Nx.tensor([[1, 2], [3, 4]])) ==
               Nx.tensor([[0.25, 0.25], [0.25, 0.25]])
    end

    defn grad_log_sum_0_sin_sum(t),
      do: grad(t, &(&1 |> Nx.log() |> Nx.sum(axes: [0]) |> Nx.sin() |> Nx.sum()))

    defn grad_log_sum_1_sin_sum(t),
      do: grad(t, &(&1 |> Nx.log() |> Nx.sum(axes: [1]) |> Nx.sin() |> Nx.sum()))

    test "computes log + sum(axis) + sin + sum" do
      lhs = grad_log_sum_0_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [0.18345697474330172, -0.33410075509515635, -0.3228698817445151],
          [0.04586424368582543, -0.13364030203806254, -0.16143494087225754]
        ])

      assert_all_close(lhs, rhs)

      lhs = grad_log_sum_1_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [-0.21916944995978982, -0.10958472497989491, -0.07305648331992994],
          [0.01875804509762369, 0.015006436078098952, 0.012505363398415794]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_log_sum_keep_sin_sum(t),
      do: grad(t, &(&1 |> Nx.log() |> Nx.sum(axes: [1], keep_axes: true) |> Nx.sin() |> Nx.sum()))

    test "computes log + sum(keep_axes) + sin + sum" do
      lhs = grad_log_sum_keep_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [-0.21916944995978982, -0.10958472497989491, -0.07305648331992994],
          [0.01875804509762369, 0.015006436078098952, 0.012505363398415794]
        ])

      assert_all_close(lhs, rhs)
    end

    defn grad_sum_0_mean(t), do: grad(t, &(&1 |> Nx.sum(axes: [0]) |> Nx.mean()))
    defn grad_sum_1_mean(t), do: grad(t, &(&1 |> Nx.sum(axes: [1]) |> Nx.mean()))

    test "computes sum(axis) + mean" do
      assert grad_sum_0_mean(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
               ])

      assert grad_sum_1_mean(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    end

    defn grad_mean_0_sum(t), do: grad(t, &(&1 |> Nx.mean(axes: [0]) |> Nx.sum()))
    defn grad_mean_1_sum(t), do: grad(t, &(&1 |> Nx.mean(axes: [1]) |> Nx.sum()))

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
      do: grad(t, &(&1 |> Nx.log() |> Nx.reshape({3, 2}) |> Nx.mean(axes: [0]) |> Nx.sum()))

    defn grad_reshape_mean_1_sum(t),
      do: grad(t, &(&1 |> Nx.log() |> Nx.reshape({3, 2}) |> Nx.mean(axes: [1]) |> Nx.sum()))

    test "computes log + reshape + mean(axis) + sum" do
      assert grad_reshape_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.16666666666666666, 0.1111111111111111],
                 [0.08333333333333333, 0.06666666666666667, 0.05555555555555555]
               ])

      assert_all_close(
        grad_reshape_mean_0_sum(~VEC[1 2i 3 1 -2i -1]),
        ~VEC[0.3333333 -0.166666i 0.111111 0.3333333 0.1666666i -0.333333]
      )

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
      do: grad(t, &(&1 |> Nx.log() |> Nx.transpose() |> Nx.mean(axes: [0]) |> Nx.sum()))

    defn grad_transpose_mean_1_sum(t),
      do: grad(t, &(&1 |> Nx.log() |> Nx.transpose() |> Nx.mean(axes: [1]) |> Nx.sum()))

    test "computes log + transpose + mean(axis) + sum" do
      assert grad_transpose_mean_0_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([
                 [0.3333333333333333, 0.16666666666666666, 0.1111111111111111],
                 [0.08333333333333333, 0.06666666666666667, 0.05555555555555555]
               ])

      assert_all_close(
        grad_transpose_mean_0_sum(~MAT[1 2i 3 1 -2i -1]),
        ~MAT[0.1666 -0.0833i 0.0555 0.1666 0.0833i -0.1666]
      )

      assert grad_transpose_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.25, 0.16666666666666666], [0.125, 0.1, 0.08333333333333333]])
    end
  end

  describe "clip" do
    defn grad_sum_clip(t), do: grad(t, &Nx.sum(Nx.clip(&1, Nx.tensor(1.0), Nx.tensor(4.0))))

    defn grad_sum_clip_wrt_to_lower_lim(t, lim), do: grad(lim, &Nx.sum(Nx.clip(t, &1, 10)))
    defn grad_sum_clip_wrt_to_upper_lim(t, lim), do: grad(lim, &Nx.sum(Nx.clip(t, -10, &1)))

    test "computes gradient with sum" do
      assert grad_sum_clip(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    end

    test "computes gradient wrt to lower lim" do
      assert grad_sum_clip_wrt_to_lower_lim(Nx.iota({5}), 2.5) == Nx.tensor(3.0)
    end

    test "computes gradient wrt to upper lim" do
      assert grad_sum_clip_wrt_to_upper_lim(Nx.iota({5}), 2.5) == Nx.tensor(2.0)
    end
  end

  describe "reduce_max rule" do
    defn grad_reduce_max(t), do: grad(t, &Nx.reduce_max(Nx.cos(Nx.exp(&1))))
    defn grad_sum_reduce_max(t), do: grad(t, &Nx.sum(Nx.reduce_max(&1, axes: [1])))
    defn grad_sum_reduce_max_cos(t), do: grad(t, &Nx.sum(Nx.reduce_max(Nx.cos(&1), axes: [1])))
    defn grad_reduce_max_sum(t), do: grad(t, &Nx.reduce_max(Nx.sum(&1, axes: [1])))
    defn grad_reduce_max_min(t), do: grad(t, &Nx.reduce_max(Nx.reduce_min(&1, axes: [0])))

    defn grad_reduce_max_min_sum(t),
      do: grad(t, &Nx.reduce_max(Nx.reduce_min(Nx.sum(&1, axes: [1]), axes: [0])))

    test "computes gradient" do
      lhs = grad_reduce_max(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))
      rhs = Nx.tensor([[0.0, -3.302372203078941, 0.0, 0.0], [-3.302372203078941, 0.0, 0.0, 0.0]])
      assert_all_close(lhs, rhs)
    end

    test "computes gradient with sum" do
      lhs = grad_sum_reduce_max(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))
      rhs = Nx.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
      assert_all_close(lhs, rhs)
    end

    test "computes gradient with sum+cos" do
      lhs = grad_sum_reduce_max_cos(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))

      rhs =
        Nx.tensor([
          [-0.8414709848078965, 0.0, 0.0, 0.0],
          [0.0, -0.42073549240394825, 0.0, -0.42073549240394825]
        ])

      assert_all_close(lhs, rhs)
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

      assert_all_close(lhs, rhs)
    end
  end

  describe "reduce_min rule" do
    defn grad_reduce_min(t), do: grad(t, &Nx.reduce_min(Nx.cos(Nx.exp(&1))))
    defn grad_sum_reduce_min(t), do: grad(t, &Nx.sum(Nx.reduce_min(&1, axes: [1])))
    defn grad_sum_reduce_min_cos(t), do: grad(t, &Nx.sum(Nx.reduce_min(Nx.cos(&1), axes: [1])))
    defn grad_reduce_min_sum(t), do: grad(t, &Nx.reduce_min(Nx.sum(&1, axes: [1])))
    defn grad_reduce_min_max(t), do: grad(t, &Nx.reduce_min(Nx.reduce_max(&1, axes: [0])))

    defn grad_reduce_min_max_sum(t),
      do: grad(t, &Nx.reduce_min(Nx.reduce_max(Nx.sum(&1, axes: [1]), axes: [0])))

    test "computes gradient" do
      lhs = grad_reduce_min(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))

      rhs =
        Nx.tensor([
          [-0.37220643914833773, 0.0, 0.0, 0.0],
          [0.0, -0.37220643914833773, 0.0, -0.37220643914833773]
        ])

      assert_all_close(lhs, rhs)
    end

    test "computes gradient with sum" do
      lhs = grad_sum_reduce_min(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))
      rhs = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.5]])
      assert_all_close(lhs, rhs)
    end

    test "computes gradient with sum+cos" do
      lhs = grad_sum_reduce_min_cos(Nx.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 1.0]]))

      rhs =
        Nx.tensor([[0.0, 0.0, -0.1411200080598672, 0.0], [0.0, 0.0, -0.1411200080598672, 0.0]])

      assert_all_close(lhs, rhs)
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

  describe "product" do
    defn grad_product(t, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.product(opts)
        |> Nx.sum()
      end)
    end

    defn composed_grad_product(t, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.cos()
        |> Nx.product(opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    test "computes gradient" do
      assert_all_close(
        grad_product(Nx.tensor([[[[1, 2, 3, 4], [2, 1, 3, -1]]]])),
        Nx.tensor([[[[-144, -72, -48, -36], [-72, -144, -48, 144]]]])
      )

      assert_all_close(
        grad_product(Nx.tensor([[[[1, 2, 3, 4], [2, 1, 3, -1], [3, 2, 3, 0]]]])),
        Nx.tensor([[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2592]]]])
      )
    end

    test "computes gradient with axes option" do
      assert_all_close(
        grad_product(Nx.tensor([[[[1, 2, 3, 4], [2, 1, 3, -1], [3, 2, 3, 0], [0, 2, 3, 0]]]]),
          axes: [3]
        ),
        Nx.tensor([[[[24, 12, 8, 6], [-3, -6, -2, 6], [0, 0, 0, 18], [0, 0, 0, 0]]]])
      )
    end

    test "computes gradient (keep_axes: true)" do
      assert_all_close(
        grad_product(Nx.tensor([[[[1, 2, 3, 4], [2, 1, 3, -1]]]]), keep_axes: true),
        Nx.tensor([[[[-144, -72, -48, -36], [-72, -144, -48, 144]]]])
      )
    end

    test "computes gradient with axes option (keep_axes: true)" do
      assert_all_close(
        grad_product(Nx.tensor([[[[1, 2, 3, 4], [2, 1, 3, -1]]]]), axes: [3], keep_axes: true),
        Nx.tensor([[[[24, 12, 8, 6], [-3, -6, -2, 6]]]])
      )
    end

    test "computes composed gradient" do
      assert_all_close(
        composed_grad_product(Nx.tensor([[[[1, 2, 3, 4], [2, 1, 3, -1]]]])),
        Nx.tensor([
          [
            [
              [0.0272486, -0.03822973, -0.00249401, 0.02025739],
              [-0.03822973, 0.0272486, -0.00249401, -0.0272486]
            ]
          ]
        ])
      )
    end
  end

  describe "sort" do
    defn grad_sum_sort(t) do
      grad(
        t,
        fn t ->
          t
          |> Nx.sort(direction: :desc)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_slice_sort(t) do
      grad(
        t,
        fn t ->
          t
          |> Nx.sort(direction: :desc)
          |> Nx.slice_along_axis(2, 3, axis: 0)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_slice_chain_sort(t) do
      grad(
        t,
        fn t ->
          t
          |> Nx.pow(2)
          |> Nx.sort(direction: :desc)
          |> Nx.slice_along_axis(2, 3, axis: 0)
          |> Nx.cos()
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_sort_power(t) do
      grad(
        t,
        fn t ->
          t
          |> Nx.pow(2)
          |> Nx.sort(direction: :desc)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_power_sort(t) do
      grad(
        t,
        fn t ->
          t
          |> Nx.sort(direction: :desc)
          |> Nx.pow(2)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_log_power_sort_cos(t, opts \\ []) do
      grad(
        t,
        fn t ->
          t
          |> Nx.cos()
          |> Nx.sort(opts)
          |> Nx.pow(2)
          |> Nx.log()
          |> Nx.sum()
        end
      )
    end

    test "computes gradient" do
      assert Nx.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) ==
               grad_sum_sort(Nx.tensor([4.0, 8.0, 15.0, 16.0, 23.0, 42.0]))

      assert Nx.tensor([2.0, 4.0, 6.0, 8.0]) == grad_sum_sort_power(Nx.tensor([1, 2, 3, 4]))

      assert Nx.tensor([2.0, 4.0, 6.0, 8.0]) == grad_sum_power_sort(Nx.tensor([1, 2, 3, 4]))

      assert Nx.tensor([-2.3156426, 0.2850931, 4.370079, -3.1148152]) ==
               grad_sum_log_power_sort_cos(Nx.tensor([4, 3, 2, 1]), [])
    end

    test "computes gradient with slicing" do
      t = Nx.tensor([4.0, 8.0, 15.0, 16.0, 23.0, 42.0])
      result = grad_sum_slice_sort(t)
      assert result == Nx.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    end

    test "computes gradient with slicing and chain rule" do
      t = Nx.tensor([4.0, 8.0, 15.0, 16.0, 23.0, 42.0])

      definition =
        t
        |> Nx.pow(2)
        |> Nx.sin()
        |> Nx.multiply(-2)
        |> Nx.multiply(t)

      mask = Nx.tensor([0, 1, 1, 1, 0, 0])
      expected = Nx.select(mask, definition, 0)
      assert expected == grad_sum_slice_chain_sort(t)
    end

    test "computes gradient along axis" do
      assert Nx.tensor([
               [
                 [-2.3156426, 0.2850931, 4.370079, -3.1148152],
                 [-3.1148152, 4.370079, 0.2850931, -2.3156426]
               ]
             ]) ==
               grad_sum_log_power_sort_cos(Nx.tensor([[[4, 3, 2, 1], [1, 2, 3, 4]]]), axis: 1)
    end
  end

  describe "take_along_axis" do
    defn grad_sum_take_along_axis(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.take_along_axis(i, axis: 1)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_take_along_axis_power(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.pow(2)
          |> Nx.take_along_axis(i, axis: 1)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_log_power_take_along_axis_cos(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.cos()
          |> Nx.take_along_axis(i, axis: 1)
          |> Nx.pow(2)
          |> Nx.log()
          |> Nx.sum()
        end
      )
    end

    test "computes gradient" do
      assert Nx.tensor([
               [3.0, 0.0],
               [1.0, 2.0],
               [1.0, 2.0]
             ]) ==
               grad_sum_take_along_axis(
                 Nx.tensor([
                   [0, 1],
                   [2, 3],
                   [4, 5]
                 ]),
                 Nx.tensor([
                   [0, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]
                 ])
               )

      assert Nx.tensor(
               [
                 [3.0, 0.0],
                 [1.0, 2.0],
                 [1.0, 2.0]
               ],
               type: {:c, 64}
             ) ==
               grad_sum_take_along_axis(
                 ~MAT[
                   0 1i
                   2i 3
                   4 5i
                 ],
                 Nx.tensor([
                   [0, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]
                 ])
               )

      assert Nx.tensor([
               [0.0, 0.0],
               [4.0, 12.0],
               [8.0, 20.0]
             ]) ==
               grad_sum_take_along_axis_power(
                 Nx.tensor([
                   [0, 1],
                   [2, 3],
                   [4, 5]
                 ]),
                 Nx.tensor([
                   [0, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]
                 ])
               )

      assert Nx.tensor([
               [-0.0, -0.0],
               [4.370079, 0.5701862],
               [-2.3156426, 13.522059]
             ]) ==
               grad_sum_log_power_take_along_axis_cos(
                 Nx.tensor([
                   [0, 1],
                   [2, 3],
                   [4, 5]
                 ]),
                 Nx.tensor([
                   [0, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]
                 ])
               )
    end
  end

  describe "take" do
    defn grad_sum_take(t, i, opts \\ [axis: 0]) do
      grad(
        t,
        fn t ->
          t
          |> Nx.take(i, opts)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_take_axis_1_power(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.pow(2)
          |> Nx.take(i, axis: 1)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_log_power_take_axis_1_cos(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.cos()
          |> Nx.take(i, axis: 1)
          |> Nx.pow(2)
          |> Nx.log()
          |> Nx.sum()
        end
      )
    end

    test "computes gradient" do
      assert Nx.tensor([
               [2.0, 2.0, 2.0, 2.0],
               [2.0, 2.0, 2.0, 2.0],
               [6.0, 6.0, 6.0, 6.0]
             ]) ==
               grad_sum_take(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [0, 1, 2, 2, 2],
                   [0, 1, 2, 2, 2]
                 ])
               )

      assert Nx.tensor([
               [0.0, 4.0, 24.0, 0.0],
               [16.0, 20.0, 72.0, 0.0],
               [32.0, 36.0, 120.0, 0.0]
             ]) ==
               grad_sum_take_axis_1_power(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [0, 1, 2, 2, 2],
                   [0, 1, 2, 2, 2]
                 ])
               )

      assert ~MAT[0 4i 24 -12i] ==
               grad_sum_take_axis_1_power(
                 ~MAT[0 1i 2 -3i],
                 Nx.tensor([[0, 1, 2, 2, 2, 3], [0, 1, 2, 2, 2, 3]])
               )

      assert Nx.tensor([
               [-0.0, -6.2296305, 26.220474, -0.0],
               [-4.631285, 13.522059, 3.4920743, -0.0],
               [27.198847, 1.8092626, -7.7803297, 0.0]
             ]) ==
               grad_sum_log_power_take_axis_1_cos(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [0, 1, 2, 2, 2],
                   [0, 1, 2, 2, 2]
                 ])
               )
    end

    test "works with more dimensions" do
      assert Nx.tensor([
               [3.0, 3.0, 3.0, 3.0],
               [3.0, 3.0, 3.0, 3.0],
               [3.0, 3.0, 3.0, 3.0]
             ]) ==
               grad_sum_take(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [
                     [[0], [1], [2]],
                     [[2], [1], [0]],
                     [[0], [1], [2]]
                   ]
                 ])
               )

      assert Nx.tensor([
               [1.0, 2.0, 1.0, 0.0],
               [1.0, 2.0, 1.0, 0.0],
               [1.0, 2.0, 1.0, 0.0]
             ]) ==
               grad_sum_take(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [
                     [[0], [1]],
                     [[2], [1]]
                   ]
                 ]),
                 axis: 1
               )
    end
  end

  describe "gather" do
    defn grad_sum_gather(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.gather(i)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_gather_power(t, i) do
      grad(
        t,
        fn t ->
          t
          |> Nx.pow(2)
          |> Nx.gather(i)
          |> Nx.sum()
        end
      )
    end

    defn grad_sum_log_power_gather_cos(t, i, opts \\ []) do
      grad(
        t,
        fn t ->
          t
          |> Nx.cos()
          |> Nx.gather(i, opts)
          |> Nx.pow(2)
          |> Nx.log()
          |> Nx.sum()
        end
      )
    end

    test "computes gradient" do
      assert Nx.tensor([
               [1.0, 3.0, 1.0, 0.0],
               [1.0, 1.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 3.0]
             ]) ==
               grad_sum_gather(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [
                     [[0, 0], [0, 1], [0, 2]],
                     [[2, 0], [1, 0], [0, 1]],
                     [[0, 1], [1, 1], [2, 2]],
                     [[2, 3], [2, 3], [2, 3]]
                   ]
                 ])
               )

      assert Nx.tensor([[0.0, 6.0, 4.0, 0.0], [8.0, 10.0, 0.0, 0.0], [16.0, 0.0, 20.0, 66.0]]) ==
               grad_sum_gather_power(
                 Nx.tensor([
                   [0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]
                 ]),
                 Nx.tensor([
                   [
                     [[0, 0], [0, 1], [0, 2]],
                     [[2, 0], [1, 0], [0, 1]],
                     [[0, 1], [1, 1], [2, 2]],
                     [[2, 3], [2, 3], [2, 3]]
                   ]
                 ])
               )

      assert ~MAT[
        0 2i 0 0
        8i 0 0 0
        16 0 0 0
      ] ==
               grad_sum_gather_power(
                 ~MAT[
                   0 1i 2 -3i
                   4i 5 6i 7
                   8 9i 10 11i
                 ],
                 Nx.tensor([
                   [
                     [[0, 0], [0, 1]],
                     [[2, 0], [1, 0]]
                   ]
                 ])
               )

      t =
        Nx.tensor([
          [0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 10, 11]
        ])

      i =
        Nx.tensor([
          [
            [[0, 0], [0, 1], [0, 2]],
            [[2, 0], [1, 0], [0, 1]],
            [[0, 1], [1, 1], [2, 2]],
            [[2, 3], [2, 3], [2, 3]]
          ]
        ])

      result =
        Nx.tensor([
          [-0.0, -9.34444522857666, 4.370079040527344, -0.0],
          [-2.3156425952911377, 6.7610297203063965, 0.0, -0.0],
          [13.5994234085083, -0.0, -1.2967215776443481, 1355.705078125]
        ])

      assert result == grad_sum_log_power_gather_cos(t, i)

      assert Nx.new_axis(result, 1) ==
               grad_sum_log_power_gather_cos(Nx.new_axis(t, 1), i, axes: [0, 2])
    end
  end

  describe "fft and ifft" do
    defn fft_grad(t, opts \\ []) do
      grad(t, fn t -> t |> Nx.fft(opts) |> Nx.sum() end)
    end

    defn ifft_grad(t, opts \\ []) do
      grad(t, fn t -> t |> Nx.ifft(opts) |> Nx.sum() end)
    end

    defn fft_composed_grad(t, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.cos()
        |> Nx.fft(opts)
        |> Nx.exp()
        |> Nx.sum()
      end)
    end

    defn ifft_composed_grad(t, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.cos()
        |> Nx.ifft(opts)
        |> Nx.exp()
        |> Nx.sum()
      end)
    end

    test "fft" do
      t =
        Nx.tensor([
          [5, 5, 0, 0],
          [2, 2, 2, 2],
          [0, Complex.new(0, 1), 0, 0]
        ])

      assert_all_close(
        Nx.tensor([
          [4, 0, 0, 0],
          [4, 0, 0, 0],
          [4, 0, 0, 0]
        ]),
        fft_grad(t)
      )

      assert_all_close(
        Nx.tensor([
          [8, 0, 0, 0],
          [8, 0, 0, 0],
          [8, 0, 0, 0]
        ]),
        fft_grad(t, length: 8)
      )

      assert_all_close(
        ~MAT[
          14.16124 12.1516 0 0 0 0
          -2.89999 0.737196 0.737196 0.737196 0 0
          0 -108.54785i 0 0 0 0
        ],
        fft_composed_grad(Nx.pad(t, 0, [{0, 0, 0}, {0, 2, 0}]), length: 4)
      )
    end

    test "ifft" do
      t =
        Nx.tensor([
          [5, 5, 0, 0],
          [2, 2, 2, 2],
          [0, Complex.new(0, 1), 0, 0]
        ])

      assert_all_close(
        Nx.tensor([
          [1, 0, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 0, 0]
        ]),
        ifft_grad(t)
      )

      assert_all_close(
        Nx.tensor([
          [1, 0, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 0, 0]
        ]),
        ifft_grad(t, length: 8)
      )

      assert_all_close(
        ~MAT[
          1.0896463 0.28715 0 0 0 0
          -0.8319124 0.077385 0.077385 0.077385 0 0
          0 -0.57873374i 0 0 0 0
        ],
        ifft_composed_grad(Nx.pad(t, 0, [{0, 0, 0}, {0, 2, 0}]), length: 4)
      )
    end
  end

  describe "indexed_put" do
    defn grad_indexed_put_target(t, i, u), do: grad(t, &Nx.sum(Nx.indexed_put(&1, i, u)))

    defn grad_indexed_put_target_composite(t, i, u, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.cos()
        |> Nx.indexed_put(i, u, opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn grad_indexed_put_updates(t, i, u), do: grad(u, &Nx.sum(Nx.indexed_put(t, i, &1)))

    defn grad_indexed_put_updates_composite(t, i, u, opts \\ []) do
      grad(u, fn u ->
        t
        |> Nx.indexed_put(i, Nx.cos(u), opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn grad_indexed_put_indices(t, i, u), do: grad(i, &Nx.sum(Nx.indexed_put(t, &1, u)))

    defn grad_indexed_put_indices_composite(t, i, u, opts \\ []) do
      grad(i, fn i ->
        t
        |> Nx.indexed_put(Nx.multiply(i, 2), u, opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn grad_indexed_put_simultaneous_composite(t, i, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.indexed_put(i, Nx.cos(t), opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    test "grad wrt to target" do
      t = Nx.iota({3, 4})
      i = Nx.tensor([[0, 0], [2, 2], [1, 0], [0, 1], [2, 3]])
      u = Nx.tensor([1, -1, 2, -2, 3])

      assert_all_close(
        Nx.tensor([
          [0, 0, 1, 1],
          [0, 1, 1, 1],
          [1, 1, 0, 0]
        ]),
        grad_indexed_put_target(t, i, u)
      )

      assert_all_close(
        Nx.tensor([
          [0, 0, -0.8316, -0.0774],
          [0, 0.9206, 0.1602, -0.4789],
          [-0.9789, -0.2525, 0, 0]
        ]),
        grad_indexed_put_target_composite(t, i, u)
      )

      assert_all_close(
        Nx.tensor([
          [0, 0, -0.8316, -0.0774],
          [0, 0.9206, 0.1602, -0.4789],
          [-0.9789, -0.2525, 0, 0]
        ])
        |> Nx.new_axis(1),
        grad_indexed_put_target_composite(Nx.new_axis(t, 1), i, Nx.new_axis(u, 1), axes: [0, 2])
      )
    end

    test "grad wrt to source" do
      t = Nx.iota({3, 4})
      i = Nx.tensor([[0, 0], [2, 2], [1, 0], [0, 1], [2, 3]])
      u = Nx.tensor([1, -1, 2, -2, 3])

      assert_all_close(Nx.broadcast(1, u), grad_indexed_put_updates(t, i, u))

      # u entries pass through the composite function f(x) = sin(cos(x));
      # f'(x) = cos(cos(x)) * (-sin(x))
      expected = u |> Nx.cos() |> Nx.cos() |> Nx.multiply(Nx.sin(u)) |> Nx.negate()
      assert_all_close(expected, grad_indexed_put_updates_composite(t, i, u))

      assert_all_close(
        Nx.new_axis(expected, 1),
        grad_indexed_put_updates_composite(Nx.new_axis(t, 1), i, Nx.new_axis(u, 1), axes: [0, 2])
      )
    end

    test "grad wrt to indices" do
      t = Nx.iota({3, 4})
      i = Nx.tensor([[0, 0], [2, 2], [1, 0], [0, 1], [2, 3]])
      u = Nx.tensor([1, -1, 2, -2, 3])

      assert_all_close(Nx.broadcast(0, i), grad_indexed_put_indices(t, i, u))
      assert_all_close(Nx.broadcast(0, i), grad_indexed_put_indices_composite(t, i, u))

      assert_all_close(
        Nx.broadcast(0, i),
        grad_indexed_put_indices_composite(Nx.new_axis(t, 1), i, Nx.new_axis(u, 1), axes: [0, 2])
      )
    end

    test "grad wrt to both source and target simultaneously" do
      # This isn't really a practical case, but we need to ensure it works
      t = Nx.iota({2})
      i = Nx.tensor([[0], [1]])

      # u entries pass through the composite function f(x) = sin(cos(x))
      # therefore: f'(x) = cos(cos(x)) * (-sin(x))

      expected = t |> Nx.cos() |> Nx.cos() |> Nx.multiply(Nx.sin(t)) |> Nx.negate()

      assert_all_close(expected, grad_indexed_put_simultaneous_composite(t, i))

      assert_all_close(
        Nx.new_axis(expected, 1),
        grad_indexed_put_simultaneous_composite(Nx.new_axis(t, 1), i, axes: [0])
      )
    end
  end

  describe "indexed_add" do
    defn grad_indexed_add_target(t, i, u) do
      grad(t, fn t ->
        t
        |> Nx.indexed_add(i, u)
        |> Nx.indexed_add(i, u)
        |> Nx.sum()
      end)
    end

    defn grad_indexed_add_target_composite(t, i, u, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.cos()
        |> Nx.indexed_add(i, u, opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn grad_indexed_add_updates(t, i, u) do
      grad(u, fn u ->
        t
        |> Nx.indexed_add(i, u)
        |> Nx.indexed_add(i, u)
        |> Nx.sum()
      end)
    end

    defn grad_indexed_add_updates_composite(t, i, u, opts \\ []) do
      grad(u, fn u ->
        t
        |> Nx.indexed_add(i, Nx.cos(u), opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn grad_indexed_add_indices(t, i, u), do: grad(i, &Nx.sum(Nx.indexed_add(t, &1, u)))

    defn grad_indexed_add_indices_composite(t, i, u, opts \\ []) do
      grad(i, fn i ->
        t
        |> Nx.indexed_add(Nx.multiply(i, 2), u, opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn grad_indexed_add_simultaneous_composite(t, i, opts \\ []) do
      grad(t, fn t ->
        t
        |> Nx.indexed_add(i, Nx.cos(t), opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    test "grad wrt to target" do
      t = Nx.iota({3, 4})
      i = Nx.tensor([[0, 0], [2, 2], [1, 0], [0, 1], [2, 3]])
      u = Nx.tensor([1, -1, 2, -2, 3])

      # The entries aren't overwritten, so the grad isn't killed on update
      # and f(x, y) = x + y implies f'(x, y) = f'(x) + f'(y)

      assert_all_close(
        Nx.tensor([
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]
        ]),
        grad_indexed_add_target(t, i, u)
      )

      assert_all_close(
        Nx.tensor([
          [0, -0.0932, -0.8316, -0.0774],
          [0.1684, 0.9206, 0.1602, -0.4789],
          [-0.9789, -0.2525, -0.1442, -0.9905]
        ]),
        grad_indexed_add_target_composite(t, i, u)
      )

      assert_all_close(
        Nx.tensor([
          [0, -0.0932, -0.8316, -0.0774],
          [0.1684, 0.9206, 0.1602, -0.4789],
          [-0.9789, -0.2525, -0.1442, -0.9905]
        ])
        |> Nx.new_axis(1),
        grad_indexed_add_target_composite(Nx.new_axis(t, 1), i, Nx.new_axis(u, 1), axes: [0, 2])
      )
    end

    test "grad wrt to source" do
      t = Nx.iota({3, 4})
      i = Nx.tensor([[0, 0], [2, 2], [1, 0], [0, 1], [2, 3]])
      u = Nx.tensor([1, -1, 2, -2, 3])

      assert_all_close(Nx.broadcast(2, u), grad_indexed_add_updates(t, i, u))

      # u entries pass through the composite function f(x) = sin(cos(x) + tn)
      # where tn is the entry in `t` corresponding to `x`; tn is constant w.r.t to x
      # therefore: f'(x) = cos(cos(x) + tn) * (-sin(x) + 0)

      cosx_tn = u |> Nx.cos() |> Nx.add(Nx.gather(t, i))
      expected = cosx_tn |> Nx.cos() |> Nx.multiply(Nx.sin(u)) |> Nx.negate()

      assert_all_close(expected, grad_indexed_add_updates_composite(t, i, u))

      assert_all_close(
        Nx.new_axis(expected, 1),
        grad_indexed_add_updates_composite(Nx.new_axis(t, 1), i, Nx.new_axis(u, 1), axes: [0, 2])
      )
    end

    test "grad wrt to indices" do
      t = Nx.iota({3, 4})
      i = Nx.tensor([[0, 0], [2, 2], [1, 0], [0, 1], [2, 3]])
      u = Nx.tensor([1, -1, 2, -2, 3])

      assert_all_close(Nx.broadcast(0, i), grad_indexed_add_indices(t, i, u))
      assert_all_close(Nx.broadcast(0, i), grad_indexed_add_indices_composite(t, i, u))

      assert_all_close(
        Nx.broadcast(0, i),
        grad_indexed_add_indices_composite(Nx.new_axis(t, 1), i, Nx.new_axis(u, 1), axes: [0, 2])
      )
    end

    test "grad wrt to both source and target simultaneously" do
      # This isn't really a practical case, but we need to ensure it works
      t = Nx.iota({2})
      i = Nx.tensor([[0], [1]])

      # u entries pass through the composite function f(x) = sin(cos(x) + x)
      # therefore: f'(x) = cos(cos(x) + x) * (-sin(x) + 1)

      cosx_tn = t |> Nx.cos() |> Nx.add(t)
      expected = cosx_tn |> Nx.cos() |> Nx.multiply(Nx.subtract(1, Nx.sin(t)))

      assert_all_close(expected, grad_indexed_add_simultaneous_composite(t, i))

      assert_all_close(
        Nx.new_axis(expected, 1),
        grad_indexed_add_simultaneous_composite(Nx.new_axis(t, 1), i, axes: [0])
      )
    end
  end

  describe "solve" do
    defn solve_grad_wrt_a(a, b) do
      grad(a, fn a ->
        a
        |> Nx.LinAlg.solve(b)
        |> Nx.sum()
      end)
    end

    test "computes the grad" do
      a =
        Nx.tensor([
          [1, 0, 1],
          [0, 1, 0],
          [-1, 0, 1]
        ])

      b = Nx.tensor([4, 3, 2])

      assert_all_close(
        solve_grad_wrt_a(a, b),
        Nx.tensor([
          [-1.0, -3.0, -3.0],
          [-1.0, -3.0, -3.0],
          [0.0, 0.0, 0.0]
        ])
      )
    end

    defn solve_grad_wrt_add(w, a, b) do
      grad(a, fn a ->
        w
        |> Nx.add(a)
        |> Nx.LinAlg.solve(b)
        |> Nx.sum()
      end)
    end

    test "computes the grad with operations before solve" do
      w = Nx.tensor([0, 0, 0])

      a =
        Nx.tensor([
          [1, 0, 1],
          [0, 1, 0],
          [-1, 0, 1]
        ])

      b = Nx.tensor([4, 3, 2])

      assert_all_close(
        solve_grad_wrt_add(w, a, b),
        Nx.tensor([
          [-1.0, -3.0, -3.0],
          [-1.0, -3.0, -3.0],
          [0.0, 0.0, 0.0]
        ])
      )
    end
  end

  describe "triangular_solve" do
    defn triangular_solve_grad_wrt_a(a, b, opts \\ []) do
      grad(a, fn a ->
        a
        |> Nx.LinAlg.triangular_solve(b, opts)
        |> Nx.sum()
      end)
    end

    defn triangular_solve_grad_wrt_b(a, b, opts \\ []) do
      grad(b, fn b ->
        a
        |> Nx.LinAlg.triangular_solve(b, opts)
        |> Nx.sum()
      end)
    end

    defn triangular_solve_composed_grad_wrt_a(a, b, opts \\ []) do
      grad(a, fn a ->
        a
        |> Nx.sin()
        |> Nx.LinAlg.triangular_solve(Nx.cos(b), opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    defn triangular_solve_composed_grad_wrt_b(a, b, opts \\ []) do
      grad(b, fn b ->
        a
        |> Nx.sin()
        |> Nx.LinAlg.triangular_solve(Nx.cos(b), opts)
        |> Nx.sin()
        |> Nx.sum()
      end)
    end

    test "computes the simple grad for tensor wrt a" do
      a =
        Nx.tensor([
          [1, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ])

      b = Nx.tensor([4, 3, 2])

      assert_all_close(
        triangular_solve_grad_wrt_a(a, b, lower: false),
        Nx.tensor([[-1, -1, -2], [0, 0, 0], [0, 0, 0]])
      )
    end

    test "computes the simple grad for tensor wrt b" do
      a =
        Nx.tensor([
          [1, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ])

      b = Nx.tensor([4, 3, 2])

      assert_all_close(
        triangular_solve_grad_wrt_b(a, b, lower: false),
        Nx.tensor([1, 0, 0])
      )
    end

    test "computes the composed grad for tensor wrt a" do
      a =
        Nx.tensor([
          [1, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ])

      b = Nx.tensor([4, 3, 2])

      assert_all_close(
        triangular_solve_composed_grad_wrt_a(a, b, lower: false),
        Nx.tensor([
          [-0.23642, 0.40336, 0.29251],
          [0.0, -0.06342, -0.04599],
          [0.0, 0.0, 0.03297]
        ])
      )
    end

    test "computes the composed grad for tensor wrt a lower: true" do
      a =
        Nx.tensor([
          [1, 0, 0],
          [1, 1, 0],
          [1, 1, 1]
        ])

      b = Nx.tensor([2, 3, 4])

      assert_all_close(
        triangular_solve_composed_grad_wrt_a(a, b, lower: true),
        Nx.tensor([
          [0.03297454, 0.0, 0.0],
          [-0.04599008, -0.06341801, 0.0],
          [0.29251346, 0.40336132, -0.23642266]
        ])
      )
    end

    test "computes the grad for tensor wrt a lower: true, transform_a: :transpose" do
      a =
        Nx.tensor([
          [1, 0, 0],
          [1, 1, 0],
          [1, 1, 1]
        ])

      b = Nx.tensor([2, 3, 4])

      assert_all_close(
        triangular_solve_grad_wrt_a(a, b, transform_a: :transpose, lower: true),
        Nx.tensor([
          [1, 0, 0],
          [1, 0, 0],
          [-4, 0, 0]
        ])
      )
    end

    test "computes the composed grad for tensor wrt b" do
      a =
        Nx.tensor([
          [1, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ])

      b = Nx.tensor([4, 3, 2])

      assert_all_close(
        triangular_solve_composed_grad_wrt_b(a, b, lower: false),
        Nx.tensor([0.8284839, 0.02428892, -0.11221229])
      )
    end

    test "computes the composed grad for tensor wrt a left_side: false" do
      a =
        Nx.tensor([
          [1, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ])

      b = Nx.tensor([2, 3, 4])

      assert_all_close(
        triangular_solve_composed_grad_wrt_a(a, b, left_side: false, lower: false),
        Nx.tensor([
          [0.03297454, -0.04599008, 0.29251346],
          [0.0, -0.06341801, 0.40336132],
          [0.0, 0.0, -0.23642266]
        ])
      )
    end

    test "computes the composed grad for tensor wrt b left_side: false" do
      a =
        Nx.tensor([
          [1, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ])

      b = Nx.tensor([2, 3, 4])

      assert_all_close(
        triangular_solve_composed_grad_wrt_b(a, b, left_side: false, lower: false),
        Nx.tensor([-0.11221229, 0.02428893, 0.82848394])
      )
    end
  end

  describe "not implemented" do
    defn grad_reduce(t), do: grad(t, &Nx.reduce(&1, 0, fn x, y -> x + y end))

    test "raises on reduce" do
      assert_raise ArgumentError, ~r"cannot compute gradient for Nx.reduce/4", fn ->
        grad_reduce(3)
      end
    end

    defn grad_quotient(t), do: grad(t, &Nx.quotient(&1, 2))

    test "raises on quotient" do
      assert_raise ArgumentError, ~r"cannot compute gradient for Nx.quotient/2", fn ->
        grad_quotient(2)
      end
    end

    defn grad_window_prod(t), do: grad(t, &Nx.window_product(&1, {}))

    test "raises on window_prod" do
      assert_raise ArgumentError, ~r"cannot compute gradient for Nx.window_product/3", fn ->
        grad_window_prod(2)
      end
    end
  end

  describe "window_scatter_min/max" do
    defn grad_window_scatter_max(t, source, init) do
      grad({t, source, init}, fn {t, source, init} ->
        Nx.window_scatter_max(t, source, init, {2, 3}, strides: [2, 3], padding: :valid)
      end)
    end

    defn grad_window_scatter_min(t, source, init) do
      grad({t, source, init}, fn {t, source, init} ->
        Nx.window_scatter_min(t, source, init, {2, 3}, strides: [2, 3], padding: :valid)
      end)
    end

    defn grad_composed_window_scatter_min(t, source, init) do
      grad({t, source, init}, fn {t, source, init} ->
        t
        |> Nx.pow(2)
        |> Nx.window_scatter_min(source, init, {2, 3}, strides: [2, 3], padding: :valid)
        |> Nx.exp()
        |> Nx.add(Nx.multiply(init, 2))
      end)
    end

    defn grad_composed_window_scatter_max(t, source, init) do
      grad({t, source, init}, fn {t, source, init} ->
        t
        |> Nx.pow(2)
        |> Nx.window_scatter_max(source, init, {2, 3}, strides: [2, 3], padding: :valid)
        |> Nx.exp()
        |> Nx.add(Nx.multiply(init, 2))
      end)
    end

    test "window_scatter_max" do
      t =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      source = Nx.tensor([[2, 6], [3, 1]])
      init_value = 0

      assert {input_grad, source_grad, init_value_grad} =
               grad_window_scatter_max(t, source, init_value)

      assert input_grad == Nx.broadcast(0, t)
      assert source_grad == Nx.broadcast(1.0, source)
      assert init_value_grad == Nx.tensor(24.0)
    end

    test "window_scatter_min" do
      t =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      source = Nx.tensor([[2, 6], [3, 1]])
      init_value = 0

      assert {input_grad, source_grad, init_value_grad} =
               grad_window_scatter_min(t, source, init_value)

      assert input_grad == Nx.broadcast(0, t)
      assert source_grad == Nx.broadcast(1.0, source)
      assert init_value_grad == Nx.tensor(24.0)
    end

    test "grad_composed_window_scatter_max" do
      t =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      source = Nx.tensor([[2, 6], [3, 1]])
      init_value = 0

      expected_source_grad =
        Nx.Defn.grad(source, fn s ->
          Nx.exp(s)
        end)

      assert {input_grad, source_grad, init_value_grad} =
               grad_composed_window_scatter_max(t, source, init_value)

      assert input_grad == Nx.broadcast(0.0, t)
      assert source_grad == expected_source_grad

      # the acummulated grad is 2 * 24 + 20 1s for the scattered values + the source grad for the positions where source was added
      assert init_value_grad == Nx.tensor(2 * 24 + 20) |> Nx.add(Nx.sum(expected_source_grad))
    end

    test "grad_composed_window_scatter_min" do
      t =
        Nx.tensor([
          [7, 2, 5, 3, 10, 2],
          [3, 8, 9, 3, 4, 2],
          [1, 5, 7, 5, 6, 1],
          [0, 6, 2, 7, 2, 8]
        ])

      source = Nx.tensor([[2, 6], [3, 1]])
      init_value = 0

      expected_source_grad =
        Nx.Defn.grad(source, fn s ->
          Nx.exp(s)
        end)

      assert {input_grad, source_grad, init_value_grad} =
               grad_composed_window_scatter_min(t, source, init_value)

      assert input_grad == Nx.broadcast(0.0, t)
      assert source_grad == expected_source_grad

      # the acummulated grad is 2 * 24 + 20 1s for the scattered values + the source grad for the positions where source was added
      assert init_value_grad == Nx.tensor(2 * 24 + 20) |> Nx.add(Nx.sum(expected_source_grad))
    end
  end

  describe "vectorization" do
    test "supports combination of vectorized and non-vectorized tensors" do
      x = Nx.tensor([[1, 2, 3], [4, 5, 6]]) |> Nx.vectorize(:x)
      y = 1

      grad = Nx.Defn.grad(y, fn y -> Nx.add(x, y) end)

      assert grad == Nx.tensor([3.0, 3.0]) |> Nx.vectorize([:x])
    end

    test "supports combination of vectorized and non-vectorized tensors over composed function" do
      x = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y]) |> Nx.vectorize(:x)
      y = 1

      grad = Nx.Defn.grad(y, fn y -> Nx.add(y, Nx.sin(x)) end)
      assert grad == Nx.tensor([3.0, 3.0]) |> Nx.vectorize([:x])

      grad = Nx.Defn.grad(x, fn x -> Nx.add(y, Nx.sin(x)) end)
      assert grad == Nx.cos(x)
    end

    # Skipping this as it's not supported yet.
    @tag :skip
    test "edge case where the same name changes meaning" do
      x = Nx.tensor([[1], [2], [3]]) |> Nx.vectorize(x: 3)

      grad =
        Nx.Defn.grad(x, fn t ->
          devec = Nx.devectorize(t, keep_names: true)
          new_axis = Nx.reshape(devec, {1, 3, 1}, names: [:x, nil, nil])

          Nx.vectorize(new_axis, x: 1)
        end)

      assert grad == Nx.tensor([[1], [1], [1]]) |> Nx.vectorize(x: 3)
    end

    test "supports heterogenous vectorization combinations" do
      x = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      y = Nx.tensor([10, 20])

      # first case: y is vectorized scalar, x is vectorized vectors, different vectorized axis names
      # expected result: equivalent to fully broadcasting one tensor onto the other
      x_vec = Nx.vectorize(x, :x)
      y_vec = Nx.vectorize(y, :y)

      grad_fun = fn x, y ->
        Nx.Defn.grad({x, y}, fn {a, b} -> Nx.multiply(a, b) end)
      end

      {grad_x_vec, grad_y_vec} = grad_fun.(x_vec, y_vec)

      # Explicit assertion on the results
      assert grad_x_vec ==
               Nx.tensor([
                 [
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0]
                 ],
                 [
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0]
                 ]
               ])
               |> Nx.vectorize([:x, :y])

      assert grad_y_vec ==
               Nx.tensor([
                 [6.0, 6.0],
                 [15.0, 15.0]
               ])
               |> Nx.vectorize([:x, :y])

      # Conceptual assertion: the result should be equivalent to calling Nx.Defn.grad with
      # each cross-entry of the combined vectors [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]

      {x0y0_wrt_x, x0y0_wrt_y} = grad_fun.(x[0], y[0])
      {x0y1_wrt_x, x0y1_wrt_y} = grad_fun.(x[0], y[1])
      {x1y0_wrt_x, x1y0_wrt_y} = grad_fun.(x[1], y[0])
      {x1y1_wrt_x, x1y1_wrt_y} = grad_fun.(x[1], y[1])

      assert grad_x_vec ==
               [x0y0_wrt_x, x0y1_wrt_x, x1y0_wrt_x, x1y1_wrt_x]
               |> Nx.stack()
               |> Nx.reshape({2, 2, 3})
               |> Nx.vectorize([:x, :y])

      assert grad_y_vec ==
               [x0y0_wrt_y, x0y1_wrt_y, x1y0_wrt_y, x1y1_wrt_y]
               |> Nx.stack()
               |> Nx.reshape({2, 2})
               |> Nx.vectorize([:x, :y])

      # second case: y is vectorized scalar, x is vectorized vectors, same vectorized axis name
      # expected result: equivalent to "row-wise" broadcasting
      x_vec = Nx.vectorize(x, :x)
      y_vec = Nx.vectorize(y, :x)
      {grad_x_vec, grad_y_vec} = Nx.Defn.grad({x_vec, y_vec}, fn {a, b} -> Nx.multiply(a, b) end)

      assert grad_x_vec ==
               Nx.tensor([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
               |> Nx.vectorize(x_vec.vectorized_axes)

      assert grad_y_vec == Nx.tensor([6.0, 15.0]) |> Nx.vectorize(y_vec.vectorized_axes)
    end
  end
end
