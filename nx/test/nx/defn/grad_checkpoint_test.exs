defmodule Nx.Defn.GradCheckpointTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Defn.Kernel, only: [grad_checkpoint: 2]
  import Nx.Testing, only: [assert_equal: 2, assert_all_close: 2]

  describe "forward pass (outside grad)" do
    defn checkpoint_computation(x) do
      grad_checkpoint(x, fn x -> Nx.sin(Nx.add(x, 1.0)) end)
    end

    defn no_checkpoint_computation(x) do
      Nx.sin(Nx.add(x, 1.0))
    end

    test "produces identical result to non-checkpointed computation" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert_equal(checkpoint_computation(x), no_checkpoint_computation(x))
    end
  end

  describe "gradient correctness" do
    defn grad_with_checkpoint(x) do
      grad(x, fn x ->
        grad_checkpoint(x, fn x -> Nx.sin(x) end)
      end)
    end

    defn grad_without_checkpoint(x) do
      grad(x, fn x -> Nx.sin(x) end)
    end

    test "simple elementwise: sin" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert_equal(grad_with_checkpoint(x), grad_without_checkpoint(x))
    end
  end

  describe "nested checkpoints" do
    defn grad_nested_checkpoint(x) do
      grad(x, fn x ->
        grad_checkpoint(x, fn x ->
          y = Nx.multiply(x, x)

          grad_checkpoint(y, fn y ->
            Nx.sum(Nx.sin(y))
          end)
        end)
      end)
    end

    defn grad_no_nested(x) do
      grad(x, fn x -> Nx.sum(Nx.sin(Nx.multiply(x, x))) end)
    end

    test "nested checkpoints produce correct gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_nested_checkpoint(x), grad_no_nested(x))
    end
  end

  describe "interaction with cond" do
    defn grad_checkpoint_with_cond(x) do
      grad(x, fn x ->
        grad_checkpoint(x, fn x ->
          if Nx.greater(Nx.sum(x), 0) do
            Nx.sum(Nx.sin(x))
          else
            Nx.sum(Nx.cos(x))
          end
        end)
      end)
    end

    defn grad_cond_no_checkpoint(x) do
      grad(x, fn x ->
        if Nx.greater(Nx.sum(x), 0) do
          Nx.sum(Nx.sin(x))
        else
          Nx.sum(Nx.cos(x))
        end
      end)
    end

    test "checkpoint with cond, either branch" do
      for x <- [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([-1.0, -2.0, -3.0])] do
        assert_equal(grad_checkpoint_with_cond(x), grad_cond_no_checkpoint(x))
      end
    end

    defn grad_cond_with_checkpoint(x) do
      grad(x, fn x ->
        if Nx.greater(Nx.sum(x), 0) do
          grad_checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end)
        else
          Nx.sum(Nx.cos(x))
        end
      end)
    end

    test "checkpoint inside a cond branch" do
      assert_equal(
        grad_cond_with_checkpoint(Nx.tensor([1.0, 2.0, 3.0])),
        grad_cond_no_checkpoint(Nx.tensor([1.0, 2.0, 3.0]))
      )

      assert_equal(
        grad_cond_with_checkpoint(Nx.tensor([-1.0, -2.0, -3.0])),
        grad_cond_no_checkpoint(Nx.tensor([-1.0, -2.0, -3.0]))
      )
    end
  end

  describe "interaction with while" do
    defn grad_checkpoint_inside_while(x) do
      grad(x, fn x ->
        {_i, acc} =
          while {i = 0, acc = x}, Nx.less(i, 3) do
            {i + 1, grad_checkpoint(acc, fn acc -> Nx.sin(acc) * acc end)}
          end

        Nx.sum(acc)
      end)
    end

    defn grad_inside_while_plain(x) do
      grad(x, fn x ->
        {_i, acc} =
          while {i = 0, acc = x}, Nx.less(i, 3) do
            {i + 1, Nx.sin(acc) * acc}
          end

        Nx.sum(acc)
      end)
    end

    test "checkpoint inside a while body" do
      x = Nx.tensor([0.5, 1.0])
      assert_equal(grad_checkpoint_inside_while(x), grad_inside_while_plain(x))
    end

    defn grad_checkpoint_with_while(x) do
      grad(x, fn x ->
        grad_checkpoint(x, fn x ->
          {_i, acc} =
            while {i = 0, acc = x}, Nx.less(i, 3) do
              {i + 1, Nx.sin(acc)}
            end

          Nx.sum(acc)
        end)
      end)
    end

    defn grad_while_no_checkpoint(x) do
      grad(x, fn x ->
        {_i, acc} =
          while {i = 0, acc = x}, Nx.less(i, 3) do
            {i + 1, Nx.sin(acc)}
          end

        Nx.sum(acc)
      end)
    end

    test "checkpoint wrapping while produces correct gradient" do
      x = Nx.tensor([0.5, 1.0])
      assert_equal(grad_checkpoint_with_while(x), grad_while_no_checkpoint(x))
    end
  end

  describe "interaction with custom_grad" do
    defn my_relu(x) do
      custom_grad(
        Nx.max(x, 0),
        [x],
        fn g -> [Nx.select(Nx.greater(x, 0), g, 0.0)] end
      )
    end

    defn grad_checkpoint_custom_grad(x) do
      grad(x, fn x ->
        grad_checkpoint(x, fn x ->
          Nx.sum(my_relu(x))
        end)
      end)
    end

    defn grad_custom_grad_no_checkpoint(x) do
      grad(x, fn x -> Nx.sum(my_relu(x)) end)
    end

    test "checkpoint with custom_grad produces correct gradient" do
      x = Nx.tensor([-1.0, 0.5, 2.0])
      assert_equal(grad_checkpoint_custom_grad(x), grad_custom_grad_no_checkpoint(x))
    end
  end

  describe "container support" do
    defn mixed_output_loss(x) do
      {value, mask} = grad_checkpoint(x, fn x -> {Nx.sin(x), x > 0} end)
      Nx.sum(value * mask)
    end

    defn mixed_output_loss_plain(x) do
      Nx.sum(Nx.sin(x) * (x > 0))
    end

    test "a boolean output used as a mask does not require a gradient" do
      x = Nx.tensor([-0.7, 0.3, 0.7])
      expected = Nx.select(Nx.greater(x, 0), Nx.cos(x), 0)

      assert_all_close(grad(x, &mixed_output_loss_plain/1), expected)
      assert_all_close(grad(x, &mixed_output_loss/1), expected)
    end

    defn duplicated_outputs(x) do
      grad_checkpoint(x, fn x ->
        y = x * x
        {y, y}
      end)
    end

    test "duplicated outputs accumulate their gradient contributions" do
      x = Nx.tensor([2.0, 3.0])

      gradient =
        grad(x, fn x ->
          {a, b} = duplicated_outputs(x)
          Nx.sum(Nx.add(a, Nx.multiply(3, b)))
        end)

      assert_equal(gradient, Nx.tensor([16.0, 24.0]))
    end

    test "either duplicated output can be used without the other" do
      x = Nx.tensor([2.0, 3.0])

      for index <- [0, 1] do
        gradient = grad(x, fn x -> Nx.sum(elem(duplicated_outputs(x), index)) end)
        assert_equal(gradient, Nx.tensor([4.0, 6.0]))
      end
    end

    test "raises when the body returns a container other than a tuple" do
      x = Nx.tensor([1.0, 2.0])

      assert_raise ArgumentError, ~r/must return a tensor or a tuple of tensors/, fn ->
        Nx.Defn.jit(fn x -> grad_checkpoint(x, fn x -> %{value: x} end) end).(x)
      end
    end

    defn grad_checkpoint_nested_container(x, y) do
      grad({x, y}, fn {x, y} ->
        {a, b, c} =
          grad_checkpoint({x, %{pair: {y, x}}}, fn {x, %{pair: {y, x_again}}} ->
            {Nx.sin(x), Nx.cos(y), x_again * y}
          end)

        Nx.sum(a * b + c)
      end)
    end

    defn grad_nested_container_plain(x, y) do
      grad({x, y}, fn {x, y} -> Nx.sum(Nx.sin(x) * Nx.cos(y) + x * y) end)
    end

    test "nested container as input" do
      x = Nx.tensor([0.5, 1.0])
      y = Nx.tensor([1.5, 2.0])
      assert_equal(grad_checkpoint_nested_container(x, y), grad_nested_container_plain(x, y))
    end
  end

  describe "edge cases" do
    defn grad_checkpoint_number_input(x) do
      grad(x, fn x ->
        Nx.sum(grad_checkpoint([x, 3.0], fn x, scale -> Nx.multiply(x, scale) end))
      end)
    end

    test "a number among the inputs is passed through to the body" do
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_checkpoint_number_input(x), Nx.tensor([3.0, 3.0]))
    end

    defn grad_checkpoint_same_input_twice(x) do
      grad(x, fn x -> Nx.sum(grad_checkpoint([x, x], fn a, b -> Nx.multiply(a, b) end)) end)
    end

    test "the same tensor passed as two inputs sums both gradients" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_same_input_twice(x), Nx.multiply(x, 2.0))
    end

    defn grad_checkpoint_unused_output(x) do
      grad(x, fn x ->
        _unused = grad_checkpoint(x, fn x -> Nx.exp(x) end)
        Nx.sum(Nx.sin(x))
      end)
    end

    test "an unused checkpoint output contributes nothing" do
      x = Nx.tensor([0.5, 1.0])
      assert_equal(grad_checkpoint_unused_output(x), Nx.cos(x))
    end

    test "raises when the function arity does not match the inputs" do
      x = Nx.tensor([1.0, 2.0])

      assert_raise ArgumentError, ~r/expected a function of arity 2 for 2 input\(s\)/, fn ->
        Nx.Defn.jit(fn x -> grad_checkpoint([x, x], fn x -> x end) end).(x)
      end
    end

    test "inputs without tensors call the function directly" do
      assert grad_checkpoint([], fn -> :no_tensors end) == :no_tensors
    end

    test "more inputs than a function can take as arguments" do
      count = 300
      weights = List.to_tuple(Enum.map(1..count, &Nx.tensor(&1 * 1.0)))

      sum_all = fn weights ->
        grad_checkpoint(weights, fn weights ->
          weights |> Tuple.to_list() |> Nx.stack() |> Nx.sum()
        end)
      end

      assert_equal(Nx.Defn.jit(sum_all).(weights), Nx.tensor(count * (count + 1) / 2))

      gradient = Nx.Defn.jit(fn weights -> grad(weights, sum_all) end).(weights)
      assert tuple_size(gradient) == count
      assert_equal(elem(gradient, 0), Nx.tensor(1.0))
      assert_equal(elem(gradient, count - 1), Nx.tensor(1.0))
    end

    defn grad_checkpoint_closed_over_opts(x, opts \\ []) do
      opts = keyword!(opts, scale: 2.0)
      offset = 1.0

      grad(x, fn x ->
        grad_checkpoint(x, fn x -> Nx.sum(Nx.multiply(x, opts[:scale]) + offset) end)
      end)
    end

    test "non-tensor values from the enclosing scope stay closed over" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_closed_over_opts(x, scale: 3.0), Nx.tensor([3.0, 3.0, 3.0]))
    end
  end

  describe "tensors read from the enclosing scope" do
    defn captured_weight_loss(x, w) do
      Nx.sum(grad_checkpoint(x, fn x -> Nx.sin(Nx.dot(x, w)) end))
    end

    test "raises in the forward pass" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0]])
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])

      assert_raise RuntimeError, ~r/expressions come from different contexts/, fn ->
        captured_weight_loss(x, w)
      end
    end

    defn grad_captured_weight(x, w), do: grad(w, &captured_weight_loss(x, &1))

    test "raises inside grad" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0]])
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])

      assert_raise RuntimeError, ~r/expressions come from different contexts/, fn ->
        grad_captured_weight(x, w)
      end
    end
  end

  describe "gradient w.r.t. weights" do
    defn grad_shared_weight_two_checkpoints(w, x) do
      grad(w, fn w ->
        first = grad_checkpoint([x, w], fn x, w -> Nx.dot(x, w) end)
        second = grad_checkpoint([first, w], fn first, w -> Nx.dot(first, w) end)
        Nx.sum(second)
      end)
    end

    defn grad_shared_weight_plain(w, x) do
      grad(w, fn w -> Nx.sum(Nx.dot(Nx.dot(x, w), w)) end)
    end

    test "a weight shared by two checkpoints accumulates both gradients" do
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      x = Nx.tensor([1.0, 2.0])
      assert_all_close(grad_shared_weight_two_checkpoints(w, x), grad_shared_weight_plain(w, x))
    end
  end

  describe "higher-order gradients" do
    defn conditional_power(x) do
      if x > 0, do: x * x, else: x * x * x
    end

    defn nested_conditional_loss(x) do
      y =
        grad_checkpoint(x, fn x ->
          y = grad_checkpoint(x * x, &conditional_power/1)
          y * y + conditional_power(x)
        end)

      y * y
    end

    @tag timeout: 5_000
    test "second-order gradient through nested checkpoints with conditionals" do
      # Bound memory use if the nested gradient expands repeatedly during evaluation.
      Process.flag(:max_heap_size, %{size: 8_000_000, kill: true, error_logger: false})

      second_derivative = grad(grad(&nested_conditional_loss/1))

      # For x > 0 the loss is (x^8 + x^2)^2, whose second derivative is
      # 240*x^14 + 180*x^8 + 12*x^2.
      assert_equal(second_derivative.(Nx.tensor(0.5)), Nx.tensor(3.7177734375))
    end

    defn nonlinear_conditional(x) do
      if x > 0, do: Nx.sin(x) * Nx.sin(x), else: Nx.cos(x) * Nx.cos(x) * Nx.cos(x)
    end

    defn nested_nonlinear_conditional_loss(x) do
      y =
        grad_checkpoint(x, fn x ->
          y = grad_checkpoint(Nx.sin(x), &nonlinear_conditional/1)
          Nx.sin(y) + nonlinear_conditional(x)
        end)

      Nx.sin(y)
    end

    defn nested_nonlinear_conditional_loss_without_checkpoint(x) do
      y = nonlinear_conditional(Nx.sin(x))
      Nx.sin(Nx.sin(y) + nonlinear_conditional(x))
    end

    @tag timeout: 5_000
    test "gradients through nested checkpoints with nonlinear conditionals" do
      Process.flag(:max_heap_size, %{size: 8_000_000, kill: true, error_logger: false})

      first_derivative = grad(&nested_nonlinear_conditional_loss/1)
      second_derivative = grad(first_derivative)
      expected_first = grad(&nested_nonlinear_conditional_loss_without_checkpoint/1)
      expected_second = grad(expected_first)

      # At 0.5 both conditionals take the first branch, at -2.0 both take the
      # second, and at 3.5 the outer takes the first while sin(3.5) < 0 sends
      # the inner to the second.
      for input <- [Nx.tensor(0.5), Nx.tensor(-2.0), Nx.tensor(3.5)] do
        assert_all_close(first_derivative.(input), expected_first.(input))
        assert_all_close(second_derivative.(input), expected_second.(input))
      end
    end

    defn grad_of_grad_outer_loss(x) do
      grad(x, fn x ->
        grad(x, fn x -> Nx.sum(Nx.sin(grad_checkpoint(x, fn x -> Nx.sin(x) end))) end)
        |> Nx.sum()
      end)
    end

    defn grad_of_grad_outer_loss_plain(x) do
      grad(x, fn x ->
        grad(x, fn x -> Nx.sum(Nx.sin(Nx.sin(x))) end)
        |> Nx.sum()
      end)
    end

    test "second-order gradient when the loss depends on the checkpoint output" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_all_close(grad_of_grad_outer_loss(x), grad_of_grad_outer_loss_plain(x))
    end
  end

  describe "vectorized input" do
    defn grad_checkpoint_vectorized_weight(x, w) do
      grad(w, fn w -> Nx.sum(grad_checkpoint([x, w], fn x, w -> Nx.sin(Nx.dot(x, w)) end)) end)
    end

    defn grad_vectorized_weight_plain(x, w) do
      grad(w, fn w -> Nx.sum(Nx.sin(Nx.dot(x, w))) end)
    end

    test "vectorized activation with a plain weight sums the weight gradient over the batch" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]) |> Nx.vectorize(:batch)
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])

      gradient = grad_checkpoint_vectorized_weight(x, w)
      assert gradient.vectorized_axes == []
      assert_all_close(gradient, grad_vectorized_weight_plain(x, w))
    end

    test "matches the gradient of each entry taken separately" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
      vectorized = Nx.vectorize(x, :batch)

      per_entry =
        for i <- 0..2 do
          grad_checkpoint_vectorized(x[i])
        end
        |> Nx.stack()
        |> Nx.vectorize(:batch)

      assert_equal(grad_checkpoint_vectorized(vectorized), per_entry)
    end

    defn grad_checkpoint_vectorized(x) do
      grad(x, fn x -> Nx.sum(grad_checkpoint(x, fn x -> Nx.sin(x) * x end)) end)
    end

    defn grad_vectorized_plain(x) do
      grad(x, fn x -> Nx.sum(Nx.sin(x) * x) end)
    end

    test "forward and gradient keep the vectorized axes" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]) |> Nx.vectorize(:batch)

      forward =
        Nx.Defn.jit(fn x -> grad_checkpoint(x, fn x -> Nx.multiply(Nx.sin(x), x) end) end).(x)

      assert forward.vectorized_axes == [batch: 3]
      assert_equal(forward, Nx.multiply(Nx.sin(x), x))

      gradient = grad_checkpoint_vectorized(x)
      assert gradient.vectorized_axes == [batch: 3]
      assert_equal(gradient, grad_vectorized_plain(x))
    end

    defn grad_checkpoint_vectorized_scale(x, scale) do
      grad(scale, fn scale ->
        Nx.sum(grad_checkpoint([x, scale], fn x, scale -> Nx.multiply(Nx.sin(x), scale) end))
      end)
    end

    defn grad_vectorized_scale_plain(x, scale) do
      grad(scale, fn scale -> Nx.sum(Nx.multiply(Nx.sin(x), scale)) end)
    end

    test "gradient with respect to a vectorized input when the activation is plain" do
      x = Nx.tensor([0.5, 1.0])
      scale = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) |> Nx.vectorize(:batch)

      gradient = grad_checkpoint_vectorized_scale(x, scale)
      assert gradient.vectorized_axes == [batch: 3]
      assert_equal(gradient, grad_vectorized_scale_plain(x, scale))
    end

    defn grad_checkpoint_vectorized_constant(x) do
      grad(x, fn x ->
        scale = Nx.vectorize(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), :batch)
        Nx.sum(grad_checkpoint([x, scale], fn x, scale -> Nx.multiply(Nx.sin(x), scale) end))
      end)
    end

    defn grad_vectorized_constant_plain(x) do
      grad(x, fn x ->
        scale = Nx.vectorize(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), :batch)
        Nx.sum(Nx.multiply(Nx.sin(x), scale))
      end)
    end

    test "a vectorized constant built inside defn as a checkpoint input" do
      x = Nx.tensor([0.5, 1.0])

      gradient = grad_checkpoint_vectorized_constant(x)
      assert gradient.vectorized_axes == [batch: 3]
      assert_equal(gradient, grad_vectorized_constant_plain(x))
    end
  end

  describe "expression tree" do
    test "checkpoint node appears in debug output" do
      fun =
        Nx.Defn.jit(
          fn x ->
            grad_checkpoint(x, fn x -> Nx.sin(x) end)
          end,
          compiler: Nx.Defn.Debug
        )

      result = fun.(Nx.tensor(1.0))
      assert inspect(result) =~ "grad_checkpoint"
    end
  end

  describe "back-to-back checkpoints" do
    defn grad_back_to_back(x) do
      grad(x, fn x ->
        x
        |> grad_checkpoint(fn x -> Nx.sin(x) end)
        |> grad_checkpoint(fn x -> x end)
        |> Nx.sum()
      end)
    end

    defn grad_back_to_back_plain(x) do
      grad(x, fn x -> x |> Nx.sin() |> Nx.sum() end)
    end

    test "identity checkpoint in the middle of chain" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_back_to_back(x), grad_back_to_back_plain(x))
    end
  end

  describe "random numbers" do
    defnp dropout(x, key) do
      {mask, _key} = Nx.Random.uniform(key, shape: Nx.shape(x))
      Nx.multiply(x, Nx.greater(mask, 0.5))
    end

    defn dropout_checkpoint_value_and_grad(x, key) do
      value_and_grad(x, fn x ->
        Nx.sum(grad_checkpoint([x, key], fn x, key -> dropout(x, key) end))
      end)
    end

    defn dropout_plain_value_and_grad(x, key) do
      value_and_grad(x, fn x -> Nx.sum(dropout(x, key)) end)
    end

    test "a random mask from an explicit key is the same in the recomputed body" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      key = Nx.Random.key(42)

      {value, gradient} = dropout_checkpoint_value_and_grad(x, key)
      {plain_value, plain_gradient} = dropout_plain_value_and_grad(x, key)
      assert_equal(value, plain_value)
      assert_equal(gradient, plain_gradient)

      kept = Nx.to_number(Nx.sum(gradient))
      assert kept > 0 and kept < 8
    end
  end

  describe "recomputation" do
    defn grad_checkpoint_used_twice(x) do
      grad(x, fn x ->
        y = grad_checkpoint(x, fn x -> Nx.sin(x) end)
        Nx.sum(Nx.add(y, Nx.multiply(y, 2.0)))
      end)
    end

    defn grad_plain_used_twice(x) do
      grad(x, fn x ->
        y = Nx.sin(x)
        Nx.sum(Nx.add(y, Nx.multiply(y, 2.0)))
      end)
    end

    test "gradient correct when checkpoint output used by multiple downstream ops" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_used_twice(x), grad_plain_used_twice(x))
    end
  end

  # The memory saving comes from the backward pass running the body again.
  # If the recomputed body were merged with the forward one, gradients would
  # stay correct and these counts would drop to one.

  describe "execution counts" do
    defp counted_body(x, parent) do
      x
      |> Nx.exp()
      |> Nx.io_call(fn _ -> send(parent, :body_ran) end)
      |> Nx.sin()
    end

    test "the body runs once in the forward pass and once more in the backward pass" do
      parent = self()
      x = Nx.tensor([1.0, 2.0])

      # Without a checkpoint, value_and_grad runs the body once.
      Nx.Defn.value_and_grad(x, fn x -> Nx.sum(counted_body(x, parent)) end)
      assert_received :body_ran
      refute_received :body_ran

      # With a checkpoint, the backward pass runs it again.
      Nx.Defn.value_and_grad(x, fn x ->
        x |> grad_checkpoint(fn x -> counted_body(x, parent) end) |> Nx.sum()
      end)

      assert_received :body_ran
      assert_received :body_ran
      refute_received :body_ran

      # The forward pass alone runs it once however many times the output is used.
      Nx.Defn.jit_apply(
        fn x ->
          y = grad_checkpoint(x, fn x -> counted_body(x, parent) end)
          Nx.add(y, Nx.multiply(y, 2.0))
        end,
        [x]
      )

      assert_received :body_ran
      refute_received :body_ran
    end
  end
end
