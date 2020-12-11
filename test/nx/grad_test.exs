defmodule Nx.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  defmodule GradHelpers do
    import Nx.Shared

    # Check the gradient of numerical function `func`.
    #
    # You must hold the function constant on every other
    # variable with a partial application of `func`:
    #
    # check_grads(& my_function(1.0, 1.0, &1, 1.0))
    def check_grads(func, grad_func, x, eps \\ 1.0e-4) do
      est_grad = finite_differences(func, x, eps)
      comp_grad = grad_func.(x)
      approx_equal?(est_grad, comp_grad, eps)
    end

    # Determines if `lhs` approx. equals `rhs` given
    # `eps`
    #
    # TODO: defn/simplify when predicates are worked out
    def approx_equal?(lhs, rhs, eps) do
      output_type = Nx.Type.merge(lhs.type, rhs.type)
      binary = Nx.Util.to_bitstring(Nx.abs(Nx.subtract(lhs, rhs)))
      value =
        match_types [output_type] do
          <<match!(var, 0)>> = binary
          read!(var, 0)
        end

      value < eps
    end

    # Numerical method for estimating the gradient
    # of `func` with respect to `x` using the finite
    # difference `eps`
    def finite_differences(func, x, eps) do
      Nx.divide(
        Nx.subtract(
          func.(Nx.add(x, Nx.divide(eps, 2.0))),
          func.(Nx.subtract(x, Nx.divide(eps, 2.0)))
        ),
        eps
      )
    end
  end

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

  describe "check_grad" do
    defn my_function(t), do: Nx.tanh(t)
    defn grad_my_function(t), do: grad(t, Nx.tanh(t))

    test "verifies gradient" do
      assert GradHelpers.check_grads(&my_function/1, &grad_my_function/1, Nx.tensor(1.0))
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
end
