defmodule Nx.Defn.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.GradHelpers

  @iters 1..25

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
      assert_raise ArgumentError,
                   "the first argument of grad must be a tensor expression or a tuple of tensor expressions, got: :invalid",
                   fn -> grad_invalid(Nx.tensor(1)) end
    end
  end

  describe "value and grad" do
    defn vg(a, b) do
      value_and_grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.power(b, 2) end)
    end

    test "computes value and grad" do
      assert vg(1, 2) ==
               {Nx.tensor(4.761594155955764, type: {:f, 32}),
                {Nx.tensor(0.41997434161402614), Nx.tensor(4.0)}}
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
          custom_grad(Nx.cos(t), fn _ans, g ->
            [{t, g * -Nx.sin(t)}]
          end)
        end)

      {cos, custom_cos}
    end

    test "computes custom grad" do
      assert {x, x} = custom_grad_meta(Nx.tensor(1))
    end

    defn random_meta(t),
      do: grad(t, fn t -> transform(Nx.exp(t), &Nx.Defn.Expr.metadata(&1, %{oops: true})) end)

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
    defn addition_rule(t), do: Nx.tanh(Nx.tanh(Nx.add(Nx.power(t, 2), Nx.power(t, 3))))
    defn grad_addition_rule(t), do: grad(t, &addition_rule/1)

    test "computes gradient of complex rules" do
      assert grad_addition_rule(Nx.tensor(1.0)) == Nx.tensor(0.15662670135498047)

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
    defn grad_product_rule(t), do: grad(t, &product_rule/1)

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

      for _ <- @iters do
        check_grads!(
          &division_rule/1,
          &grad_division_rule/1,
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
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
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
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
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
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
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
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
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
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
          Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        )
      end
    end
  end

  describe "power rule" do
    defn power_rule(t), do: Nx.power(t, 3)
    defn grad_power_rule(t), do: grad(t, &power_rule/1)

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
    defn grad_exp_rule(t), do: grad(t, &exp_rule/1)

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
    defn grad_atan2_rule(t), do: grad(t, &atan2_rule/1)

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

    defn grad_dot_both_rule(x), do: grad(x, &Nx.sum(Nx.dot(Nx.power(&1, 2), Nx.power(&1, 3))))

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

      compare_tensors!(lhs, rhs)

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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

      lhs = grad_sum_conv_y_general_stride_lhs_dilated(x, y)
      rhs = Nx.tensor([[[[312.0, 328.0], [312.0, 328.0]]], [[[312.0, 328.0], [312.0, 328.0]]]])

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

      lhs = grad_mean_conv_y_general_stride_rhs_dilated(x, y)

      rhs =
        Nx.tensor([[[[7.4000006, 8.2], [7.4000006, 8.2]]], [[[7.4000006, 8.2], [7.4000006, 8.2]]]])

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

      lhs = grad_sum_conv_y_channels_last(x, y)

      rhs =
        Nx.tensor([
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]],
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]],
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]],
          [[[1134.0, 1188.0], [1350.0, 1404.0]], [[1161.0, 1215.0], [1377.0, 1431.0]]]
        ])

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

      lhs = grad_sum_conv_y_feature_groups(x, y)

      rhs =
        Nx.tensor([
          [[[45.0, 54.0], [81.0, 90.0]], [[189.0, 198.0], [225.0, 234.0]]],
          [[[333.0, 342.0], [369.0, 378.0]], [[477.0, 486.0], [513.0, 522.0]]]
        ])

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

      lhs = grad_sum_conv_y_batch_groups(x, y)

      rhs =
        Nx.tensor([
          [[[84.0, 60.0], [120.0, 84.0]]],
          [[[300.0, 204.0], [336.0, 228.0]]],
          [[[516.0, 348.0], [552.0, 372.0]]]
        ])

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)

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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
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

      compare_tensors!(lhs, rhs)
    end
  end

  describe "outer rule" do
    defn grad_outer_lhs_rule(x, y), do: grad(x, &Nx.sum(Nx.outer(&1, y)))

    test "computes gradient for tensors on lhs" do
      assert grad_outer_lhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
               Nx.tensor([[15.0], [15.0], [15.0]])
    end

    defn grad_outer_rhs_rule(x, y), do: grad(y, &Nx.sum(Nx.outer(x, &1)))

    test "computes gradient for tensors on rhs" do
      assert grad_outer_rhs_rule(Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[1, 2, 3, 4, 5]])) ==
               Nx.tensor([[6.0, 6.0, 6.0, 6.0, 6.0]])
    end

    defn grad_outer_both_rule(x), do: grad(x, &Nx.sum(Nx.outer(Nx.power(&1, 2), Nx.power(&1, 3))))

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
    defn grad_tanh_exp(t), do: grad(t, &Nx.tanh(Nx.exp(&1)))

    test "computes gradient" do
      assert grad_tanh_exp(Nx.tensor(1.0)) == Nx.tensor(0.046936701983213425)

      for _ <- @iters do
        t = Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
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
        t = Nx.random_uniform({}, 0.0, 10.0, type: {:f, 64})
        check_grads!(&grad_tanh_base/1, &grad_grad_tanh/1, t)
      end
    end
  end

  describe "tuples" do
    defnp tuple_pattern({a, b}), do: Nx.power(a, 2) + b
    defn grad_tuple_pattern(t), do: grad(t, &tuple_pattern({&1, 2.0}))

    test "as patterns" do
      assert grad_tuple_pattern(Nx.tensor(1.0)) == Nx.tensor(2.0)
    end

    defn grad_tuple_input(a, b) do
      grad({a, b}, fn {a, b} -> Nx.power(a, 2) * Nx.power(b, 3) end)
    end

    defn grad_tuple_input(a, b, c) do
      grad({a, b, c}, fn {a, b, c} -> Nx.power(a, 2) * Nx.power(b, 3) * Nx.power(c, 4) end)
    end

    defn grad_tuple_unused(a, b) do
      grad({a, b}, fn {a, _b} -> Nx.power(a, a) end)
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
        [:cbrt, :cos, :exp, :expm1, :log, :log1p, :logistic] ++
          [:mean, :negate, :rsqrt, :sin, :sqrt, :sum, :tanh] do
    describe "#{fun}" do
      grad_fun = :"grad_#{fun}"
      defn unquote(grad_fun)(t), do: grad(t, &(Nx.unquote(fun) / 1))

      test "computes gradient" do
        for _ <- @iters do
          t = Nx.random_uniform({}, 0.1, 10.0, type: {:f, 64})
          check_grads!(&Nx.unquote(fun)(&1), &(__MODULE__.unquote(grad_fun) / 1), t)
        end
      end
    end
  end

  describe "tan" do
    defn grad_tan(t), do: grad(t, &Nx.tan/1)

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
    defn grad_asin(t), do: grad(t, &Nx.asin/1)
    defn grad_acos(t), do: grad(t, &Nx.acos/1)
    defn grad_atan(t), do: grad(t, &Nx.atan/1)

    test "computes gradient of inverse trig functions" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -0.999, 0.999, type: {:f, 32})
        check_grads!(&Nx.asin/1, &grad_asin/1, t, atol: 1.0e-5, rtol: 1.0e-2)
        check_grads!(&Nx.acos/1, &grad_acos/1, t, atol: 0.1, rtol: 1.0e-2)
        check_grads!(&Nx.atan/1, &grad_atan/1, t, atol: 0.1, rtol: 1.0e-2)
        check_grads!(&Nx.atan/1, &grad_atan/1, Nx.multiply(1000.0, t), atol: 1.0e-2)
      end
    end
  end

  describe "hyperbolics" do
    defn grad_sinh(t), do: grad(t, &Nx.sinh/1)
    defn grad_cosh(t), do: grad(t, &Nx.cosh/1)

    test "computes gradient" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -10, 10, type: {:f, 64})
        check_grads!(&Nx.sinh/1, &grad_sinh/1, t)
        check_grads!(&Nx.cosh/1, &grad_cosh/1, t)
      end
    end

    test "computes multidimensional gradient" do
      t = Nx.random_uniform({Enum.count(@iters)}, -10, 10, type: {:f, 64})
      check_grads!(&Nx.sinh/1, &grad_sinh/1, t)
      check_grads!(&Nx.cosh/1, &grad_cosh/1, t)
    end
  end

  describe "inverse hyperbolic functions" do
    defn grad_asinh(t), do: grad(t, &Nx.asinh/1)
    defn grad_acosh(t), do: grad(t, &Nx.acosh/1)
    defn grad_atanh(t), do: grad(t, &Nx.atanh/1)

    test "computes gradient of inverse hyperbolic functions" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -100.0, 100.0, type: {:f, 64})
        check_grads!(&Nx.asinh/1, &grad_asinh/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end

      for _ <- @iters do
        t = Nx.random_uniform({}, 1.01, 100.0, type: {:f, 64})
        check_grads!(&Nx.acosh/1, &grad_acosh/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end

      for _ <- @iters do
        t = Nx.random_uniform({}, -0.999, 0.999, type: {:f, 64})
        check_grads!(&Nx.atanh/1, &grad_atanh/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end
  end

  describe "erf" do
    defn grad_erf(t), do: grad(t, &Nx.erf/1)

    test "computes the gradient" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -100.0, 100.0, type: {:f, 64})
        check_grads!(&Nx.erf/1, &grad_erf/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end
  end

  describe "erfc" do
    defn grad_erfc(t), do: grad(t, &Nx.erfc/1)

    test "computes the gradient" do
      for _ <- @iters do
        t = Nx.random_uniform({}, -100.0, 100.0, type: {:f, 64})
        check_grads!(&Nx.erfc/1, &grad_erfc/1, t, atol: 1.0e-4)
      end
    end
  end

  describe "erf_inv" do
    defn grad_erf_inv(t), do: grad(t, &Nx.erf_inv/1)

    test "computes gradient close to 0.0" do
      for _ <- @iters do
        t = Nx.random_uniform({}, 0.0, 0.9, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end

    test "computes gradient between 0.9 and 0.95" do
      for _ <- @iters do
        t = Nx.random_uniform({}, 0.9, 0.95, type: {:f, 64})
        check_grads!(&Nx.erf_inv/1, &grad_erf_inv/1, t, atol: 1.0e-5, rtol: 1.0e-2)
      end
    end

    test "computes gradient between 0.95 and 0.98" do
      for _ <- @iters do
        t = Nx.random_uniform({}, 0.95, 0.98, type: {:f, 64})
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
        assert_in_delta(Nx.to_scalar(grad_erf_inv(x)), y, 0.01)
      end
    end
  end

  describe "broadcast" do
    defn grad_sum_broadcast(t), do: grad(t, &Nx.sum(Nx.broadcast(&1, {3, 2, 2})))

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

  describe "concatenate" do
    defn concatenate_grad(t) do
      grad(t, &Nx.sum(Nx.concatenate([&1, &1], axis: 0)))
    end

    defn concatenate_grad_power(t) do
      grad(
        t,
        &Nx.sum(Nx.concatenate([Nx.power(&1, 2), Nx.power(&1, 3)], axis: 0))
      )
    end

    defn concatenate_grad_composed(t) do
      grad(
        t,
        &Nx.sum(Nx.concatenate([Nx.log(Nx.power(&1, 2)), Nx.log(Nx.power(&1, 3))], axis: 3))
      )
    end

    defn concatenate_grad_log1p(t) do
      grad(
        t,
        &Nx.sum(Nx.log1p(Nx.concatenate([Nx.power(&1, 2), Nx.power(&1, 3)], axis: 3)))
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
                 [-0.13259541988372803, -0.14328323304653168, -0.02223709039390087],
                 [0.13743554055690765, 0.16928502917289734, 0.06221092492341995]
               ])
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

      assert grad_sum_dynamic_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

      lhs = grad_sum_pad_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

      rhs =
        Nx.tensor([
          [0.9905542274249228, -0.7629358670030943, -1.8149862437674833],
          [-1.1983466382499552, 1.520047340015915, 1.7603121921923377]
        ])

      compare_tensors!(lhs, rhs)

      lhs = grad_sum_pad_dynamic_slice(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

      compare_tensors!(lhs, rhs)
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
    defn grad_mean_put_slice_operand(t1, t2), do: grad(t1, &Nx.mean(Nx.put_slice(&1, t2, [0, 1])))
    defn grad_mean_put_slice_update(t1, t2), do: grad(t2, &Nx.mean(Nx.put_slice(t1, &1, [0, 1])))

    defn grad_sum_pad_put_slice_cos_operand(t1, t2) do
      grad(t1, fn t ->
        t
        |> Nx.cos()
        |> Nx.put_slice(Nx.sin(t2), [1, 2])
        |> Nx.pad(Nx.mean(Nx.sin(t)), [{2, 1, 2}, {-1, 2, 0}])
        |> Nx.sum()
      end)
    end

    defn grad_sum_pad_put_slice_sin_update(t1, t2) do
      grad(t2, fn t ->
        t1
        |> Nx.cos()
        |> Nx.put_slice(Nx.sin(t), [1, 2])
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

      compare_tensors!(lhs, rhs)

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

      compare_tensors!(lhs, rhs)
    end

    test "computes gradient of update" do
      lhs =
        grad_mean_put_slice_update(
          Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
          Nx.tensor([[10.0, 11.0]])
        )

      rhs = Nx.tensor([[0.16666667, 0.16666667]])

      compare_tensors!(lhs, rhs)

      lhs =
        grad_sum_pad_put_slice_sin_update(
          Nx.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 10.0]]),
          Nx.tensor([[10.0, 11.0], [12.0, 13.0]])
        )

      rhs = Nx.tensor([[-0.83907153, 0.0044257], [0.84385396, 0.90744678]])

      compare_tensors!(lhs, rhs)
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
    defn grad_abs(t), do: grad(t, &Nx.sum(Nx.abs(&1)))

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
    defn grad_max(t), do: grad(t, &Nx.sum(Nx.max(Nx.power(&1, 2), Nx.power(&1, 3))))

    test "computes gradient with tensors" do
      assert grad_max(Nx.tensor([[1.0], [2.0], [3.0]])) == Nx.tensor([[2.5], [12.0], [27.0]])

      assert grad_max(Nx.tensor([[1.25, 2.5, 2.75], [1.0, 4.0, 6.0], [2.0, 3.0, 2.0]])) ==
               Nx.tensor([[4.6875, 18.75, 22.6875], [2.5, 48.0, 108.0], [12.0, 27.0, 12.0]])
    end
  end

  describe "min" do
    defn grad_min(t), do: grad(t, &Nx.sum(Nx.min(Nx.power(&1, 2), Nx.power(&1, 3))))

    test "computes gradient with tensors" do
      assert grad_min(Nx.tensor([[1.0], [2.0], [3.0]])) == Nx.tensor([[2.5], [4.0], [6.0]])

      assert grad_min(Nx.tensor([[1.25, 2.5, 2.75], [1.0, 4.0, 6.0], [2.0, 3.0, 2.0]])) ==
               Nx.tensor([[2.5, 5.0, 5.5], [2.5, 8.0, 12.0], [4.0, 6.0, 4.0]])
    end
  end

  describe "select rule" do
    defn grad_sum_select(t),
      do: grad(t, &Nx.sum(Nx.select(Nx.greater(&1, 0.0), Nx.exp(&1), Nx.cos(&1))))

    defn grad_max_select(t),
      do: grad(t, &Nx.reduce_max(Nx.select(Nx.greater(&1, 0.0), Nx.exp(&1), Nx.cos(&1))))

    defn grad_sum_select_sum(t),
      do:
        grad(
          t,
          &Nx.sum(Nx.select(Nx.greater(Nx.sum(&1, axes: [0]), 0.0), Nx.sum(&1, axes: [0]), 0.0))
        )

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

      compare_tensors!(lhs, rhs)
    end

    test "computes the gradient with max+select" do
      lhs = grad_max_select(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]]))
      rhs = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 148.4131591025766, 0.0]])

      compare_tensors!(lhs, rhs)
    end

    test "computes the gradient with sum+select+sum" do
      lhs =
        grad_sum_select_sum(Nx.tensor([[-2.0, 1.0, 0.0, 3.0, -3.0], [1.0, 2.0, 0.0, 5.0, -1.0]]))

      rhs = Nx.tensor([[0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]])

      compare_tensors!(lhs, rhs)
    end
  end

  describe "as_type/bitcast" do
    defn grad_as_type(t), do: grad(t, &Nx.sum(Nx.as_type(&1, {:f, 32})))
    defn grad_bitcast(t), do: grad(t, &Nx.sum(Nx.bitcast(&1, {:f, 32})))

    test "as_type passes through" do
      assert grad_as_type(Nx.tensor([1, 2, 3])) == Nx.tensor([1.0, 1.0, 1.0])
    end

    test "bitcast passes through" do
      assert grad_bitcast(Nx.tensor([1, 2, 3], type: {:s, 32})) == Nx.tensor([1.0, 1.0, 1.0])
    end
  end

  describe "if" do
    defn grad_if(t), do: grad(t, fn t -> if(t + 1, do: Nx.power(t, 2), else: Nx.power(t, 3)) end)

    defn grad_sum_if(t) do
      grad(t, fn t -> Nx.sum(if(Nx.all?(t), do: Nx.power(t, 2), else: Nx.power(t, 3))) end)
    end

    defn grad_if_sum(t) do
      grad(t, fn t -> if(Nx.all?(t), do: Nx.sum(Nx.power(t, 2)), else: Nx.sum(Nx.power(t, 3))) end)
    end

    defn grad_if_tuple(t) do
      grad(t, fn t ->
        {{a, b}, c} =
          if t + 1 do
            {{Nx.power(t, 2), Nx.power(t, 3)}, Nx.power(t, 4)}
          else
            {{Nx.power(t, 4), Nx.power(t, 3)}, Nx.power(t, 2)}
          end

        a * b + c
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
      assert grad_if_tuple(Nx.tensor(-1)) == Nx.tensor(5.0)
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
      do: grad(t, &(&1 |> Nx.log() |> Nx.sum(axes: [1], keep_axes: true) |> Nx.sin() |> Nx.sum()))

    test "computes log + sum(keep_axes) + sin + sum" do
      lhs = grad_log_sum_keep_sin_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]]))

      rhs =
        Nx.tensor([
          [-0.21916944995978982, -0.10958472497989491, -0.07305648331992994],
          [0.01875804509762369, 0.015006436078098952, 0.012505363398415794]
        ])

      compare_tensors!(lhs, rhs)
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

      assert grad_transpose_mean_1_sum(Nx.tensor([[1, 2, 3], [4, 5, 6]])) ==
               Nx.tensor([[0.5, 0.25, 0.16666666666666666], [0.125, 0.1, 0.08333333333333333]])
    end
  end

  describe "clip" do
    defn grad_sum_clip(t), do: grad(t, &Nx.sum(Nx.clip(&1, Nx.tensor(1.0), Nx.tensor(4.0))))

    test "computes gradient with sum" do
      assert grad_sum_clip(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) ==
               Nx.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
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

  describe "reduce product rule" do
    defn grad_reduce_product(t), do: grad(t, &Nx.product/1)

    test "computes the gradient with product" do
      lhs = grad_reduce_product(Nx.tensor([[1.0, 4.0, 2.0], [3.0, 6.0, 6.0]]))
      rhs = Nx.tensor([[864.0, 216.0, 432.0], [288.0, 144.0, 144.0]])
      compare_tensors!(lhs, rhs)
    end

    defn grad_sum_grad_product(t), do: grad(t, fn t -> Nx.product(grad(t, &Nx.product/1)) end)

    test "computes the second order" do
      lhs = grad_sum_grad_product(Nx.tensor([[0.95, 0.90, 0.95], [0.75, 0.74, 1.0]]))

      rhs =
        Nx.tensor([[0.09798507, 0.10342869, 0.09798507], [0.12411442, 0.12579165, 0.09308582]])

      compare_tensors!(lhs, rhs)
    end

    defn grad_reduce_product_cos(t), do: grad(t, &Nx.product(Nx.power(&1, 2)))

    test "computes the gradient with product of inner function" do
      lhs = grad_reduce_product_cos(Nx.iota({3, 1, 2, 2}))
      rhs = Nx.broadcast(0.0, {3, 1, 2, 2})
      compare_tensors!(lhs, rhs)
    end

    defn grad_reduce_product_sum(t), do: grad(t, &Nx.product(Nx.sum(&1, axes: [1])))

    test "computes the gradient with product of sum" do
      x = Nx.multiply(0.1, Nx.iota({3, 2, 2, 2}, type: {:f, 32}))
      lhs = grad_reduce_product_sum(x)

      rhs =
        Nx.tensor([
          [
            [[3028.823, 2019.2152], [1514.4116, 1211.5292]],
            [[3028.823, 2019.2152], [1514.4116, 1211.5292]]
          ],
          [
            [[605.76465, 550.6951], [504.8038, 465.97278]],
            [[605.76465, 550.6951], [504.8038, 465.97278]]
          ],
          [
            [[336.53592, 318.82346], [302.88232, 288.45935]],
            [[336.53592, 318.82346], [302.88232, 288.45935]]
          ]
        ])

      assert Nx.all?(Nx.less_equal(Nx.abs(Nx.subtract(lhs, rhs)), 1.0e-3)) ==
               Nx.tensor(1, type: {:u, 8})
    end

    defn grad_reduce_product_product(t), do: grad(t, &Nx.product(Nx.product(&1, axes: [3])))

    test "computes the gradient with product of product" do
      x = Nx.iota({1, 3, 2, 2}, type: {:f, 32})
      lhs = grad_reduce_product_product(x)

      rhs =
        Nx.tensor([
          [
            [[39_916_800.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]]
          ]
        ])

      compare_tensors!(lhs, rhs)
    end

    defn grad_reduce_product_min(t), do: grad(t, &Nx.product(Nx.reduce_min(&1, axes: [1, 2])))

    test "computes the gradient with product of min" do
      x = Nx.iota({3, 2, 2, 3}, type: {:f, 32})
      lhs = grad_reduce_product_min(x)

      rhs =
        Nx.tensor([
          [[[68_140_800.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
          [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
          [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ])

      compare_tensors!(lhs, rhs)
    end

    defn grad_reduce_product_multiple_axes(t), do: grad(t, &Nx.sum(Nx.product(&1, axes: [0, 3])))

    test "computes the gradient for product with multiple axes" do
      x = Nx.multiply(Nx.iota({2, 2, 2, 2}, type: {:f, 32}), 0.5)
      lhs = grad_reduce_product_multiple_axes(x)

      rhs =
        Nx.tensor([
          [[[9.0, 0.0], [41.25, 27.5]], [[97.5, 78.0], [183.75, 157.5]]],
          [[[0.0, 0.0], [8.25, 7.5]], [[32.5, 30.0], [78.75, 73.5]]]
        ])

      compare_tensors!(lhs, rhs)
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

  defp compare_tensors!(left, right) do
    atol = 1.0e-7
    rtol = 1.0e-4

    try do
      assert Nx.all_close?(left, right, atol: atol, rtol: rtol) == Nx.tensor(1, type: {:u, 8})
    rescue
      # So we can see the diff
      _ -> assert left == right
    end
  end
end
