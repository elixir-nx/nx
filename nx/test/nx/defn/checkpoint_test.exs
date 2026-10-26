defmodule Nx.Defn.CheckpointTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Defn.Kernel, only: [checkpoint: 2]
  import Nx.Testing, only: [assert_equal: 2, assert_all_close: 2]

  # --- Forward pass: checkpoint is a no-op ---

  describe "forward pass (outside grad)" do
    defn checkpoint_identity(x) do
      checkpoint(x, fn x -> x end)
    end

    test "returns same result as calling the function directly" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(checkpoint_identity(x), x)
    end

    defn checkpoint_computation(x) do
      checkpoint(x, fn x -> Nx.sin(Nx.add(x, 1.0)) end)
    end

    defn no_checkpoint_computation(x) do
      Nx.sin(Nx.add(x, 1.0))
    end

    test "produces identical result to non-checkpointed computation" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert_equal(checkpoint_computation(x), no_checkpoint_computation(x))
    end
  end

  # --- Gradient correctness: checkpoint produces same gradients ---

  describe "gradient correctness" do
    defn grad_with_checkpoint(x) do
      grad(x, fn x ->
        checkpoint(x, fn x -> Nx.sin(x) end)
      end)
    end

    defn grad_without_checkpoint(x) do
      grad(x, fn x -> Nx.sin(x) end)
    end

    test "simple elementwise: sin" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert_equal(grad_with_checkpoint(x), grad_without_checkpoint(x))
    end

    defn grad_checkpoint_multiply(x) do
      grad(x, fn x ->
        checkpoint(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
      end)
    end

    defn grad_no_checkpoint_multiply(x) do
      grad(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
    end

    test "reduction: sum of squares" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_multiply(x), grad_no_checkpoint_multiply(x))
    end
  end

  # --- Nested checkpoints ---

  describe "nested checkpoints" do
    defn grad_nested_checkpoint(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          y = Nx.multiply(x, x)

          checkpoint(y, fn y ->
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

  # --- Interaction with control flow ---

  describe "interaction with cond" do
    defn grad_checkpoint_with_cond(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
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

    test "checkpoint with cond (true branch)" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_with_cond(x), grad_cond_no_checkpoint(x))
    end

    test "checkpoint with cond (false branch)" do
      x = Nx.tensor([-1.0, -2.0, -3.0])
      assert_equal(grad_checkpoint_with_cond(x), grad_cond_no_checkpoint(x))
    end

    defn grad_cond_with_checkpoint(x) do
      grad(x, fn x ->
        if Nx.greater(Nx.sum(x), 0) do
          checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end)
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
            {i + 1, checkpoint(acc, fn acc -> Nx.sin(acc) * acc end)}
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
        checkpoint(x, fn x ->
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

  # --- Interaction with custom_grad ---

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
        checkpoint(x, fn x ->
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

  # --- Container inputs/outputs ---

  describe "container support" do
    defn mixed_output_loss(x) do
      {value, mask} = checkpoint(x, fn x -> {Nx.sin(x), x > 0} end)
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
      checkpoint(x, fn x ->
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
        Nx.Defn.jit(fn x -> checkpoint(x, fn x -> %{value: x} end) end).(x)
      end
    end

    defn grad_checkpoint_map_input(params, x) do
      grad(params, fn params ->
        checkpoint([params, x], fn params, x ->
          Nx.sum(Nx.dot(x, params.w) + params.b)
        end)
      end)
    end

    defn grad_map_input_plain(params, x) do
      grad(params, fn params -> Nx.sum(Nx.dot(x, params.w) + params.b) end)
    end

    test "map as checkpoint input" do
      params = %{w: Nx.tensor([[0.5, -0.3], [0.2, 0.8]]), b: Nx.tensor([0.1, -0.1])}
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_checkpoint_map_input(params, x), grad_map_input_plain(params, x))
    end

    defn grad_checkpoint_nested_container(x, y) do
      grad({x, y}, fn {x, y} ->
        {a, b, c} =
          checkpoint({x, %{pair: {y, x}}}, fn {x, %{pair: {y, x_again}}} ->
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

    defn grad_checkpoint_tuple_output(x) do
      grad(x, fn x ->
        {a, b} =
          checkpoint(x, fn x ->
            {Nx.sin(x), Nx.cos(x)}
          end)

        Nx.sum(Nx.add(a, b))
      end)
    end

    defn grad_tuple_no_checkpoint(x) do
      grad(x, fn x ->
        Nx.sum(Nx.add(Nx.sin(x), Nx.cos(x)))
      end)
    end

    test "checkpoint returning tuple produces correct gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_tuple_output(x), grad_tuple_no_checkpoint(x))
    end
  end

  # --- Value and grad ---

  describe "value_and_grad with checkpoint" do
    defn vag_with_checkpoint(x) do
      value_and_grad(x, fn x ->
        x
        |> checkpoint(fn x -> Nx.sin(x) end)
        |> Nx.sum()
      end)
    end

    defn vag_without_checkpoint(x) do
      value_and_grad(x, fn x -> x |> Nx.sin() |> Nx.sum() end)
    end

    test "value_and_grad produces same value and gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      {val_cp, grad_cp} = vag_with_checkpoint(x)
      {val_no, grad_no} = vag_without_checkpoint(x)
      assert_equal(val_cp, val_no)
      assert_equal(grad_cp, grad_no)
    end
  end

  # --- Edge cases ---

  describe "edge cases" do
    defn grad_checkpoint_number_input(x) do
      grad(x, fn x -> Nx.sum(checkpoint([x, 3.0], fn x, scale -> Nx.multiply(x, scale) end)) end)
    end

    test "a number among the inputs is passed through to the body" do
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_checkpoint_number_input(x), Nx.tensor([3.0, 3.0]))
    end

    defn grad_checkpoint_constant_input(x) do
      grad(x, fn x ->
        Nx.sum(checkpoint([x, Nx.tensor([1.0, 2.0])], fn x, scale -> Nx.multiply(x, scale) end))
      end)
    end

    test "a constant tensor among the inputs" do
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_checkpoint_constant_input(x), Nx.tensor([1.0, 2.0]))
    end

    defn grad_checkpoint_same_input_twice(x) do
      grad(x, fn x -> Nx.sum(checkpoint([x, x], fn a, b -> Nx.multiply(a, b) end)) end)
    end

    test "the same tensor passed as two inputs sums both gradients" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_same_input_twice(x), Nx.multiply(x, 2.0))
    end

    defn grad_checkpoint_unused_output(x) do
      grad(x, fn x ->
        _unused = checkpoint(x, fn x -> Nx.exp(x) end)
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
        Nx.Defn.jit(fn x -> checkpoint([x, x], fn x -> x end) end).(x)
      end
    end

    test "inputs without tensors call the function directly" do
      assert checkpoint([], fn -> :no_tensors end) == :no_tensors
    end

    defn grad_checkpoint_scalar(x) do
      grad(x, fn x ->
        checkpoint(x, fn x -> Nx.multiply(x, x) end)
      end)
    end

    test "scalar input/output" do
      x = Nx.tensor(3.0)
      assert_equal(grad_checkpoint_scalar(x), Nx.tensor(6.0))
    end

    defn grad_checkpoint_no_grad_path(x, y) do
      grad(x, fn x ->
        checkpoint([x, y], fn x, y -> Nx.sum(Nx.multiply(x, y)) end)
      end)
    end

    test "input that is not a grad target" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      y = Nx.tensor([4.0, 5.0, 6.0])
      assert_equal(grad_checkpoint_no_grad_path(x, y), y)
    end

    test "more inputs than a function can take as arguments" do
      count = 300
      weights = List.to_tuple(Enum.map(1..count, &Nx.tensor(&1 * 1.0)))

      sum_all = fn weights ->
        checkpoint(weights, fn weights -> weights |> Tuple.to_list() |> Nx.stack() |> Nx.sum() end)
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
        checkpoint(x, fn x -> Nx.sum(Nx.multiply(x, opts[:scale]) + offset) end)
      end)
    end

    test "non-tensor values from the enclosing scope stay closed over" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_closed_over_opts(x, scale: 3.0), Nx.tensor([3.0, 3.0, 3.0]))
    end

    defn grad_checkpoint_high_rank(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          x |> Nx.sin() |> Nx.sum()
        end)
      end)
    end

    test "high-rank tensor" do
      x = Nx.iota({2, 3, 4}, type: :f32)
      expected = Nx.Defn.grad(x, &Nx.sum(Nx.sin(&1)))
      assert_equal(grad_checkpoint_high_rank(x), expected)
    end
  end

  # --- Tensors read from the enclosing scope ---

  describe "tensors read from the enclosing scope" do
    defn captured_weight_loss(x, w) do
      Nx.sum(checkpoint(x, fn x -> Nx.sin(Nx.dot(x, w)) end))
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

  # --- Gradient w.r.t. weights (training use case) ---

  describe "gradient w.r.t. weights" do
    defn grad_shared_weight_two_checkpoints(w, x) do
      grad(w, fn w ->
        first = checkpoint([x, w], fn x, w -> Nx.dot(x, w) end)
        second = checkpoint([first, w], fn first, w -> Nx.dot(first, w) end)
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

    defn grad_of_grad_weight_checkpoint(w, x) do
      grad(w, fn w ->
        grad(w, fn w -> Nx.sum(checkpoint([x, w], fn x, w -> Nx.sin(Nx.dot(x, w)) end)) end)
        |> Nx.sum()
      end)
    end

    defn grad_of_grad_weight_plain(w, x) do
      grad(w, fn w -> grad(w, fn w -> Nx.sum(Nx.sin(Nx.dot(x, w))) end) |> Nx.sum() end)
    end

    test "second-order gradient with respect to a weight" do
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      x = Nx.tensor([1.0, 2.0])
      assert_all_close(grad_of_grad_weight_checkpoint(w, x), grad_of_grad_weight_plain(w, x))
    end

    defn dense_layer(x, w) do
      Nx.dot(x, w) |> Nx.max(0)
    end

    defn grad_weights_with_checkpoint(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} ->
        hidden = checkpoint([x, w1], fn x, w1 -> dense_layer(x, w1) end)
        output = checkpoint([hidden, w2], fn hidden, w2 -> dense_layer(hidden, w2) end)
        Nx.sum(output)
      end)
    end

    defn grad_weights_no_checkpoint(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} ->
        x
        |> dense_layer(w1)
        |> dense_layer(w2)
        |> Nx.sum()
      end)
    end

    test "gradient flows to weights passed as inputs" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      x = Nx.tensor([1.0, 2.0])

      assert_equal(grad_weights_with_checkpoint(w1, w2, x), grad_weights_no_checkpoint(w1, w2, x))
    end

    defn vag_params_with_checkpoint(params, x) do
      value_and_grad(params, fn params ->
        hidden = checkpoint([x, params.w1], fn x, w1 -> dense_layer(x, w1) end)
        output = checkpoint([hidden, params.w2], fn hidden, w2 -> dense_layer(hidden, w2) end)
        Nx.sum(output)
      end)
    end

    defn vag_params_no_checkpoint(params, x) do
      value_and_grad(params, fn params ->
        x |> dense_layer(params.w1) |> dense_layer(params.w2) |> Nx.sum()
      end)
    end

    test "value_and_grad w.r.t. map of params (training pattern)" do
      params = %{
        w1: Nx.tensor([[0.5, -0.3], [0.2, 0.8]]),
        w2: Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      }

      x = Nx.tensor([1.0, 2.0])

      {val_cp, grad_cp} = vag_params_with_checkpoint(params, x)
      {val_no, grad_no} = vag_params_no_checkpoint(params, x)
      assert_equal(val_cp, val_no)
      assert_equal(grad_cp, grad_no)
    end
  end

  # --- Diamond/shared input pattern ---

  describe "shared input (diamond pattern)" do
    defn grad_diamond_checkpoint(x) do
      grad(x, fn x ->
        a = checkpoint(x, fn x -> Nx.sin(x) end)
        b = checkpoint(x, fn x -> Nx.cos(x) end)
        Nx.sum(Nx.multiply(a, b))
      end)
    end

    defn grad_diamond_no_checkpoint(x) do
      grad(x, fn x ->
        Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(x)))
      end)
    end

    test "two checkpoints sharing the same input" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_diamond_checkpoint(x), grad_diamond_no_checkpoint(x))
    end
  end

  # --- Checkpoint in the middle of a chain ---

  describe "partial checkpointing" do
    defn grad_middle_checkpoint(x) do
      grad(x, fn x ->
        x
        |> Nx.multiply(2.0)
        |> checkpoint(fn x -> Nx.sin(Nx.exp(x)) end)
        |> Nx.sum()
      end)
    end

    defn grad_middle_no_checkpoint(x) do
      grad(x, fn x ->
        x |> Nx.multiply(2.0) |> Nx.exp() |> Nx.sin() |> Nx.sum()
      end)
    end

    test "ops before and after checkpoint boundary" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_equal(grad_middle_checkpoint(x), grad_middle_no_checkpoint(x))
    end
  end

  # --- stop_grad interaction ---

  describe "interaction with stop_grad" do
    defn grad_checkpoint_with_stop_grad(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.multiply(x, stop_grad(Nx.sin(x))))
        end)
      end)
    end

    defn grad_stop_grad_no_checkpoint(x) do
      grad(x, fn x ->
        Nx.sum(Nx.multiply(x, stop_grad(Nx.sin(x))))
      end)
    end

    test "stop_grad inside checkpoint" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_checkpoint_with_stop_grad(x), grad_stop_grad_no_checkpoint(x))
    end
  end

  # --- Higher-order gradients ---

  describe "higher-order gradients" do
    defn conditional_power(x) do
      if x > 0, do: x * x, else: x * x * x
    end

    defn nested_conditional_loss(x) do
      y =
        checkpoint(x, fn x ->
          y = checkpoint(x * x, &conditional_power/1)
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
        checkpoint(x, fn x ->
          y = checkpoint(Nx.sin(x), &nonlinear_conditional/1)
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

    defn grad_of_grad_checkpoint(x) do
      grad(x, fn x ->
        grad(x, fn x ->
          checkpoint(x, fn x -> Nx.sum(Nx.pow(x, 3)) end)
        end)
        |> Nx.sum()
      end)
    end

    defn grad_of_grad_no_checkpoint(x) do
      grad(x, fn x ->
        grad(x, fn x ->
          Nx.sum(Nx.pow(x, 3))
        end)
        |> Nx.sum()
      end)
    end

    test "second-order gradient through checkpoint" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(grad_of_grad_checkpoint(x), grad_of_grad_no_checkpoint(x))
    end

    defn grad_of_grad_outer_loss(x) do
      grad(x, fn x ->
        grad(x, fn x -> Nx.sum(Nx.sin(checkpoint(x, fn x -> Nx.sin(x) end))) end)
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

  # --- Additional edge cases ---

  describe "zero gradient through checkpoint" do
    defn grad_checkpoint_constant(x) do
      grad(x, fn x ->
        checkpoint(x, fn _x -> Nx.tensor(42.0) end)
      end)
    end

    test "checkpoint returning constant gives zero gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = grad_checkpoint_constant(x)
      assert_equal(result, Nx.broadcast(0.0, {3}))
    end
  end

  describe "vectorized input" do
    defn grad_checkpoint_vectorized_weight(x, w) do
      grad(w, fn w -> Nx.sum(checkpoint([x, w], fn x, w -> Nx.sin(Nx.dot(x, w)) end)) end)
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

    defn grad_checkpoint_vectorized_intermediate(x) do
      grad(x, fn x ->
        hidden = Nx.cos(x)
        Nx.sum(checkpoint(hidden, fn hidden -> Nx.sin(hidden) * hidden end))
      end)
    end

    defn grad_vectorized_intermediate_plain(x) do
      grad(x, fn x ->
        hidden = Nx.cos(x)
        Nx.sum(Nx.sin(hidden) * hidden)
      end)
    end

    test "vectorized intermediate as the checkpoint input" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0]]) |> Nx.vectorize(:batch)

      assert_equal(
        grad_checkpoint_vectorized_intermediate(x),
        grad_vectorized_intermediate_plain(x)
      )
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
      grad(x, fn x -> Nx.sum(checkpoint(x, fn x -> Nx.sin(x) * x end)) end)
    end

    defn grad_vectorized_plain(x) do
      grad(x, fn x -> Nx.sum(Nx.sin(x) * x) end)
    end

    test "forward and gradient keep the vectorized axes" do
      x = Nx.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]) |> Nx.vectorize(:batch)

      forward = Nx.Defn.jit(fn x -> checkpoint(x, fn x -> Nx.multiply(Nx.sin(x), x) end) end).(x)
      assert forward.vectorized_axes == [batch: 3]
      assert_equal(forward, Nx.multiply(Nx.sin(x), x))

      gradient = grad_checkpoint_vectorized(x)
      assert gradient.vectorized_axes == [batch: 3]
      assert_equal(gradient, grad_vectorized_plain(x))
    end

    defn grad_checkpoint_vectorized_scale(x, scale) do
      grad(scale, fn scale ->
        Nx.sum(checkpoint([x, scale], fn x, scale -> Nx.multiply(Nx.sin(x), scale) end))
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
        Nx.sum(checkpoint([x, scale], fn x, scale -> Nx.multiply(Nx.sin(x), scale) end))
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

  # --- Expression tree inspection ---

  describe "expression tree" do
    test "checkpoint node appears in debug output" do
      fun =
        Nx.Defn.jit(
          fn x ->
            checkpoint(x, fn x -> Nx.sin(x) end)
          end,
          compiler: Nx.Defn.Debug
        )

      result = fun.(Nx.tensor(1.0))
      assert inspect(result) =~ "checkpoint"
    end
  end

  # --- Input/output shape mismatch ---

  describe "shape-changing checkpoint" do
    defn grad_shape_change(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(x, axes: [1])
        end)
        |> Nx.sum()
      end)
    end

    defn grad_shape_change_plain(x) do
      grad(x, fn x ->
        x |> Nx.sum(axes: [1]) |> Nx.sum()
      end)
    end

    test "input shape differs from output shape" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert_equal(grad_shape_change(x), grad_shape_change_plain(x))
    end
  end

  # --- Deep nesting: checkpoint inside while inside checkpoint ---

  describe "deep nesting" do
    defn grad_checkpoint_while_checkpoint(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          {_i, acc} =
            while {i = 0, acc = x}, Nx.less(i, 2) do
              {i + 1, checkpoint(acc, fn acc -> Nx.sin(acc) end)}
            end

          Nx.sum(acc)
        end)
      end)
    end

    defn grad_deep_plain(x) do
      grad(x, fn x ->
        {_i, acc} =
          while {i = 0, acc = x}, Nx.less(i, 2) do
            {i + 1, Nx.sin(acc)}
          end

        Nx.sum(acc)
      end)
    end

    test "checkpoint inside while inside checkpoint" do
      x = Nx.tensor([0.5, 1.0])
      assert_equal(grad_checkpoint_while_checkpoint(x), grad_deep_plain(x))
    end
  end

  # --- Back-to-back checkpoints with no ops between ---

  describe "back-to-back checkpoints" do
    defn grad_back_to_back(x) do
      grad(x, fn x ->
        x
        |> checkpoint(fn x -> Nx.sin(x) end)
        |> checkpoint(fn x -> x end)
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

  # --- Matmul / dot patterns (neural network ops) ---

  describe "neural network op patterns" do
    defn grad_checkpoint_matmul_chain(w1, w2, w3, x) do
      grad(x, fn x ->
        first = checkpoint([x, w1], fn x, w1 -> Nx.dot(x, w1) |> Nx.max(0) end)
        second = checkpoint([first, w2], fn first, w2 -> Nx.dot(first, w2) |> Nx.max(0) end)
        output = checkpoint([second, w3], fn second, w3 -> Nx.dot(second, w3) end)
        Nx.sum(output)
      end)
    end

    defn grad_matmul_chain_plain(w1, w2, w3, x) do
      grad(x, fn x ->
        x
        |> Nx.dot(w1)
        |> Nx.max(0)
        |> Nx.dot(w2)
        |> Nx.max(0)
        |> Nx.dot(w3)
        |> Nx.sum()
      end)
    end

    test "3-layer matmul-relu chain" do
      w1 = Nx.tensor([[0.5, -0.3, 0.1], [0.2, 0.8, -0.4]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3], [0.5, -0.1]])
      w3 = Nx.tensor([[0.3], [-0.5]])
      x = Nx.tensor([1.0, 2.0])

      assert_equal(
        grad_checkpoint_matmul_chain(w1, w2, w3, x),
        grad_matmul_chain_plain(w1, w2, w3, x)
      )
    end

    defnp dropout(x, key) do
      {mask, _key} = Nx.Random.uniform(key, shape: Nx.shape(x))
      Nx.multiply(x, Nx.greater(mask, 0.5))
    end

    defn dropout_checkpoint_value_and_grad(x, key) do
      value_and_grad(x, fn x -> Nx.sum(checkpoint([x, key], fn x, key -> dropout(x, key) end)) end)
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

  # --- Recomputation behavior ---

  describe "recomputation" do
    defn grad_checkpoint_used_twice(x) do
      grad(x, fn x ->
        y = checkpoint(x, fn x -> Nx.sin(x) end)
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

  # --- Rematerialization proxy: body execution counts ---
  #
  # Peak-memory itself is not directly observable in unit tests, but the
  # mechanism that produces the memory saving is: the backward pass must
  # RE-EXECUTE the checkpointed body instead of reusing the forward pass's
  # intermediates. We observe executions with an io_call side effect inside
  # the body. If a future change (or, on EXLA, common-subexpression
  # elimination without an optimization barrier) dedupes the rematerialized
  # tree against the forward one, these counts silently drop back to 1 and
  # the memory benefit is gone even though every gradient stays correct.

  describe "rematerialization execution counts" do
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
        x |> checkpoint(fn x -> counted_body(x, parent) end) |> Nx.sum()
      end)

      assert_received :body_ran
      assert_received :body_ran
      refute_received :body_ran

      # The forward pass alone runs it once however many times the output is used.
      Nx.Defn.jit_apply(
        fn x ->
          y = checkpoint(x, fn x -> counted_body(x, parent) end)
          Nx.add(y, Nx.multiply(y, 2.0))
        end,
        [x]
      )

      assert_received :body_ran
      refute_received :body_ran
    end
  end
end
