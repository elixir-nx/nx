defmodule Nx.Defn.CheckpointTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Defn.Kernel, only: [checkpoint: 1, checkpoint: 2]
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

    defn checkpoint_chain(x) do
      x
      |> checkpoint(fn x -> Nx.multiply(x, 2.0) end)
      |> checkpoint(fn x -> Nx.add(x, 1.0) end)
      |> checkpoint(fn x -> Nx.pow(x, 2) end)
    end

    defn no_checkpoint_chain(x) do
      x |> Nx.multiply(2.0) |> Nx.add(1.0) |> Nx.pow(2)
    end

    test "chained checkpoints produce identical result" do
      x = Nx.tensor(3.0)
      assert_equal(checkpoint_chain(x), no_checkpoint_chain(x))
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

    defn grad_checkpoint_composed(x) do
      grad(x, fn x ->
        x
        |> checkpoint(fn x -> Nx.tanh(x) end)
        |> Nx.sum()
      end)
    end

    defn grad_no_checkpoint_composed(x) do
      grad(x, fn x -> x |> Nx.tanh() |> Nx.sum() end)
    end

    test "composed: tanh then sum" do
      x = Nx.tensor([0.5, 1.0, 2.0])
      assert_equal(grad_checkpoint_composed(x), grad_no_checkpoint_composed(x))
    end
  end

  # --- Multi-layer chain (the primary use case) ---

  describe "multi-layer checkpoint" do
    defn dense_block(x, w) do
      Nx.dot(x, w) |> Nx.max(0)
    end

    defn grad_checkpointed_layers(w1, w2, x) do
      grad(x, fn x ->
        hidden = checkpoint([x, w1], fn x, w1 -> dense_block(x, w1) end)
        output = checkpoint([hidden, w2], fn hidden, w2 -> dense_block(hidden, w2) end)
        Nx.sum(output)
      end)
    end

    defn grad_plain_layers(w1, w2, x) do
      grad(x, fn x ->
        x
        |> dense_block(w1)
        |> dense_block(w2)
        |> Nx.sum()
      end)
    end

    test "dense layers produce same gradient with and without checkpoint" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      x = Nx.tensor([1.0, 2.0])

      assert_equal(grad_checkpointed_layers(w1, w2, x), grad_plain_layers(w1, w2, x))
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

  describe "captured tensors" do
    defn dense(x, w), do: Nx.dot(x, w) |> Nx.max(0)

    defn forward_captured_weight(x, w) do
      checkpoint(x, fn x -> dense(x, w) end)
    end

    test "a captured weight is read by the forward pass" do
      x = Nx.tensor([1.0, 2.0])
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      assert_equal(forward_captured_weight(x, w), dense(x, w))
    end

    defn grad_captured_weights(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} ->
        x
        |> checkpoint(fn x -> dense(x, w1) end)
        |> checkpoint(fn x -> dense(x, w2) end)
        |> Nx.sum()
      end)
    end

    defn grad_weights_plain(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} -> x |> dense(w1) |> dense(w2) |> Nx.sum() end)
    end

    test "the gradient flows to weights captured by the body" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_captured_weights(w1, w2, x), grad_weights_plain(w1, w2, x))
    end

    defn grad_captured_params(params, x) do
      value_and_grad(params, fn params ->
        x
        |> checkpoint(fn x -> dense(x, params.w1) end)
        |> checkpoint(fn x -> dense(x, params.w2) end)
        |> Nx.sum()
      end)
    end

    defn grad_params_plain(params, x) do
      value_and_grad(params, fn params ->
        x |> dense(params.w1) |> dense(params.w2) |> Nx.sum()
      end)
    end

    test "a captured map of parameters" do
      params = %{
        w1: Nx.tensor([[0.5, -0.3], [0.2, 0.8]]),
        w2: Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      }

      x = Nx.tensor([1.0, 2.0])
      {value, gradient} = grad_captured_params(params, x)
      {plain_value, plain_gradient} = grad_params_plain(params, x)
      assert_equal(value, plain_value)
      assert_equal(gradient, plain_gradient)
    end

    defn grad_captured_not_target(x, y) do
      grad(x, fn x -> checkpoint(x, fn x -> Nx.sum(Nx.multiply(x, y)) end) end)
    end

    test "a captured tensor that is not a grad target" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      y = Nx.tensor([4.0, 5.0, 6.0])
      assert_equal(grad_captured_not_target(x, y), y)
    end

    defn grad_rebound_name(x, y) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          y = Nx.multiply(x, 2.0)
          Nx.sum(Nx.multiply(y, y))
        end)
      end)
    end

    test "the body may rebind a name from the enclosing scope" do
      x = Nx.tensor([1.0, 2.0])
      y = Nx.tensor([100.0, 100.0])
      assert_equal(grad_rebound_name(x, y), Nx.tensor([8.0, 16.0]))
    end

    defn grad_zero_arity(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} ->
        hidden = checkpoint(fn -> dense(x, w1) end)
        checkpoint(fn -> dense(hidden, w2) end) |> Nx.sum()
      end)
    end

    test "checkpoint/1 takes every tensor from the enclosing scope" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_zero_arity(w1, w2, x), grad_weights_plain(w1, w2, x))
    end

    test "checkpoint/1 requires an inline function" do
      assert_raise ArgumentError, ~r/expects an inline fn/, fn ->
        Code.eval_quoted(
          quote do
            require Nx.Defn.Kernel
            Nx.Defn.Kernel.checkpoint(&Nx.exp/1)
          end
        )
      end
    end

    defn grad_indexed_capture(ws, x) do
      grad(ws, fn ws -> Nx.sum(checkpoint(x, fn x -> dense(x, ws[0]) |> dense(ws[1]) end)) end)
    end

    defn grad_indexed_plain(ws, x) do
      grad(ws, fn ws -> Nx.sum(dense(x, ws[0]) |> dense(ws[1])) end)
    end

    test "indexing a captured variable passes it whole" do
      ws = Nx.tensor([[[0.5, -0.3], [0.2, 0.8]], [[0.1, 0.4], [-0.2, 0.3]]])
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_indexed_capture(ws, x), grad_indexed_plain(ws, x))

      expr =
        Nx.Defn.debug_expr(fn ws, x -> checkpoint(x, fn x -> dense(x, ws[0]) end) end).(ws, x)

      assert inspect(expr) =~ ~r/parameter b:0\s+f32\[2\]\[2\]\[2\]/
      assert inspect(expr) =~ "block checkpoint, a, b"
    end

    defn vectorized_scale(x, scale) do
      checkpoint(x, fn x -> Nx.multiply(Nx.sin(x), scale) end)
    end

    test "a captured vectorized tensor keeps its axes in the forward pass" do
      x = Nx.tensor([0.5, 1.0])
      scale = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) |> Nx.vectorize(:batch)

      result = vectorized_scale(x, scale)
      assert result.vectorized_axes == [batch: 3]
      assert_equal(result, Nx.multiply(Nx.sin(x), scale))
    end

    defn grad_captured_vectorized(x, scale) do
      grad(scale, fn scale -> Nx.sum(checkpoint(x, fn x -> Nx.multiply(Nx.sin(x), scale) end)) end)
    end

    defn grad_vectorized_scale_plain(x, scale) do
      grad(scale, fn scale -> Nx.sum(Nx.multiply(Nx.sin(x), scale)) end)
    end

    test "the gradient with respect to a captured vectorized tensor" do
      x = Nx.tensor([0.5, 1.0])
      scale = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) |> Nx.vectorize(:batch)

      gradient = grad_captured_vectorized(x, scale)
      assert gradient.vectorized_axes == [batch: 3]
      assert_equal(gradient, grad_vectorized_scale_plain(x, scale))
    end

    defn grad_captured_vectorized_constant(x) do
      grad(x, fn x ->
        scale = Nx.vectorize(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), :batch)
        Nx.sum(checkpoint(x, fn x -> Nx.multiply(Nx.sin(x), scale) end))
      end)
    end

    defn grad_vectorized_constant_plain(x) do
      grad(x, fn x ->
        scale = Nx.vectorize(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), :batch)
        Nx.sum(Nx.multiply(Nx.sin(x), scale))
      end)
    end

    test "a vectorized constant captured from the enclosing scope" do
      x = Nx.tensor([0.5, 1.0])
      gradient = grad_captured_vectorized_constant(x)
      assert gradient.vectorized_axes == [batch: 3]
      assert_equal(gradient, grad_vectorized_constant_plain(x))
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

  # --- Numerical precision ---

  describe "numerical precision" do
    defn grad_checkpoint_exp_log(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.log(Nx.exp(x)))
        end)
      end)
    end

    defn grad_exp_log_no_checkpoint(x) do
      grad(x, fn x -> Nx.sum(Nx.log(Nx.exp(x))) end)
    end

    test "exp/log chain produces bitwise identical gradient" do
      x = Nx.tensor([0.1, 1.0, 5.0])
      assert_equal(grad_checkpoint_exp_log(x), grad_exp_log_no_checkpoint(x))
    end
  end

  # --- Multiple outputs consumed independently ---

  describe "multiple outputs consumed separately" do
    defn grad_multi_output_checkpoint(x) do
      grad(x, fn x ->
        {a, b} =
          checkpoint(x, fn x ->
            {Nx.sin(x), Nx.cos(x)}
          end)

        Nx.sum(Nx.pow(a, 2)) + Nx.sum(Nx.pow(b, 3))
      end)
    end

    defn grad_multi_output_no_checkpoint(x) do
      grad(x, fn x ->
        a = Nx.sin(x)
        b = Nx.cos(x)
        Nx.sum(Nx.pow(a, 2)) + Nx.sum(Nx.pow(b, 3))
      end)
    end

    test "tuple outputs used in independent expressions" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert_all_close(grad_multi_output_checkpoint(x), grad_multi_output_no_checkpoint(x))
    end
  end

  # --- Many sequential checkpoints (stress test) ---

  describe "many sequential checkpoints" do
    defn apply_checkpointed_sins(x) do
      x
      |> checkpoint(&Nx.sin/1)
      |> checkpoint(&Nx.sin/1)
      |> checkpoint(&Nx.sin/1)
      |> checkpoint(&Nx.sin/1)
      |> checkpoint(&Nx.sin/1)
    end

    defn apply_plain_sins(x) do
      x |> Nx.sin() |> Nx.sin() |> Nx.sin() |> Nx.sin() |> Nx.sin()
    end

    test "many sequential checkpointed layers forward" do
      x = Nx.tensor([0.5, 1.0])
      assert_equal(apply_checkpointed_sins(x), apply_plain_sins(x))
    end

    test "gradient through many sequential checkpointed layers" do
      x = Nx.tensor([0.5, 1.0])

      grad_cp = Nx.Defn.grad(x, fn x -> Nx.sum(apply_checkpointed_sins(x)) end)
      grad_plain = Nx.Defn.grad(x, fn x -> Nx.sum(apply_plain_sins(x)) end)
      assert_equal(grad_cp, grad_plain)
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

  describe "broadcasting inside checkpoint" do
    defn grad_checkpoint_broadcast(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          w = Nx.tensor([[1.0, 2.0, 3.0]])
          Nx.sum(Nx.multiply(x, w))
        end)
      end)
    end

    defn grad_broadcast_no_checkpoint(x) do
      grad(x, fn x ->
        w = Nx.tensor([[1.0, 2.0, 3.0]])
        Nx.sum(Nx.multiply(x, w))
      end)
    end

    test "broadcasting inside checkpoint" do
      x = Nx.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
      assert_equal(grad_checkpoint_broadcast(x), grad_broadcast_no_checkpoint(x))
    end
  end

  describe "dtype preservation" do
    defn grad_checkpoint_f64(x) do
      grad(x, fn x ->
        checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end)
      end)
    end

    defn grad_f64_no_checkpoint(x) do
      grad(x, fn x -> Nx.sum(Nx.sin(x)) end)
    end

    test "f64 tensors" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f64)
      assert_equal(grad_checkpoint_f64(x), grad_f64_no_checkpoint(x))
    end

    test "bf16 tensors" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :bf16)
      result = grad_checkpoint_f64(x)
      expected = grad_f64_no_checkpoint(x)
      assert_equal(result, expected)
    end
  end

  describe "shape-changing ops inside checkpoint" do
    defn grad_checkpoint_reshape(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          x |> Nx.reshape({6}) |> Nx.sin() |> Nx.sum()
        end)
      end)
    end

    defn grad_reshape_no_checkpoint(x) do
      grad(x, fn x ->
        x |> Nx.reshape({6}) |> Nx.sin() |> Nx.sum()
      end)
    end

    test "reshape inside checkpoint" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert_equal(grad_checkpoint_reshape(x), grad_reshape_no_checkpoint(x))
    end

    defn grad_checkpoint_transpose(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          x |> Nx.transpose() |> Nx.sin() |> Nx.sum()
        end)
      end)
    end

    defn grad_transpose_no_checkpoint(x) do
      grad(x, fn x ->
        x |> Nx.transpose() |> Nx.sin() |> Nx.sum()
      end)
    end

    test "transpose inside checkpoint" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert_equal(grad_checkpoint_transpose(x), grad_transpose_no_checkpoint(x))
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
  end

  # --- JIT compilation ---

  describe "jit compilation" do
    test "checkpoint works via Nx.Defn.jit" do
      fun =
        Nx.Defn.jit(fn x ->
          checkpoint(x, fn x -> Nx.sin(x) end)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(fun.(x), Nx.sin(x))
    end

    test "grad through checkpoint via jit" do
      fun =
        Nx.Defn.jit(fn x ->
          Nx.Defn.grad(x, fn x ->
            checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end)
          end)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(fun.(x), Nx.Defn.grad(x, &Nx.sum(Nx.sin(&1))))
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

  # --- Several inputs ---

  describe "several inputs" do
    defn grad_multi_capture(w1, w2, b, x) do
      grad(x, fn x ->
        checkpoint([x, w1, w2, b], fn x, w1, w2, b ->
          x |> Nx.dot(w1) |> Nx.add(b) |> Nx.dot(w2) |> Nx.sum()
        end)
      end)
    end

    defn grad_multi_capture_plain(w1, w2, b, x) do
      grad(x, fn x ->
        x |> Nx.dot(w1) |> Nx.add(b) |> Nx.dot(w2) |> Nx.sum()
      end)
    end

    test "checkpoint with several weight matrices and a bias" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1], [0.4]])
      b = Nx.tensor([0.1, -0.1])
      x = Nx.tensor([1.0, 2.0])

      assert_equal(grad_multi_capture(w1, w2, b, x), grad_multi_capture_plain(w1, w2, b, x))
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

    defn grad_expand_shape(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.stack([x, Nx.multiply(x, 2.0)])
        end)
        |> Nx.sum()
      end)
    end

    defn grad_expand_shape_plain(x) do
      grad(x, fn x ->
        Nx.stack([x, Nx.multiply(x, 2.0)]) |> Nx.sum()
      end)
    end

    test "output larger than input" do
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_expand_shape(x), grad_expand_shape_plain(x))
    end
  end

  # --- Integer input ---

  describe "integer input" do
    defn checkpoint_integer_forward(x) do
      checkpoint(x, fn x -> Nx.add(x, 1) end)
    end

    test "integer tensor forward pass" do
      x = Nx.tensor([1, 2, 3])
      assert_equal(checkpoint_integer_forward(x), Nx.tensor([2, 3, 4]))
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

  # --- Shared function reference ---

  describe "shared function across checkpoints" do
    defn grad_shared_fun(x) do
      sin_fn = &Nx.sin/1

      grad(x, fn x ->
        x
        |> checkpoint(sin_fn)
        |> checkpoint(sin_fn)
        |> checkpoint(sin_fn)
        |> Nx.sum()
      end)
    end

    defn grad_shared_plain(x) do
      grad(x, fn x ->
        x |> Nx.sin() |> Nx.sin() |> Nx.sin() |> Nx.sum()
      end)
    end

    test "same function reused across multiple checkpoints" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert_equal(grad_shared_fun(x), grad_shared_plain(x))
    end
  end

  # --- Tuple as input ---

  describe "tuple input" do
    defn grad_tuple_input(x, y) do
      grad({x, y}, fn {x, y} ->
        {a, b} =
          checkpoint({x, y}, fn {x, y} ->
            {Nx.sin(x), Nx.cos(y)}
          end)

        Nx.sum(Nx.multiply(a, b))
      end)
    end

    defn grad_tuple_input_plain(x, y) do
      grad({x, y}, fn {x, y} ->
        Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(y)))
      end)
    end

    test "tuple as checkpoint input" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      y = Nx.tensor([0.5, 1.0, 1.5])
      assert_equal(grad_tuple_input(x, y), grad_tuple_input_plain(x, y))
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

  # --- Complex number support ---

  describe "complex tensors" do
    defn grad_checkpoint_complex(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.real(Nx.multiply(x, x)))
        end)
      end)
    end

    defn grad_complex_plain(x) do
      grad(x, fn x ->
        Nx.sum(Nx.real(Nx.multiply(x, x)))
      end)
    end

    test "complex tensor input" do
      x = Nx.tensor([Complex.new(1.0, 2.0), Complex.new(3.0, -1.0)])
      assert_equal(grad_checkpoint_complex(x), grad_complex_plain(x))
    end
  end

  # --- Slice and gather inside checkpoint ---

  describe "indexing ops inside checkpoint" do
    defn grad_checkpoint_slice(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.slice(x, [1], [2]))
        end)
      end)
    end

    defn grad_slice_plain(x) do
      grad(x, fn x -> Nx.sum(Nx.slice(x, [1], [2])) end)
    end

    test "slice inside checkpoint" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      assert_equal(grad_checkpoint_slice(x), grad_slice_plain(x))
    end

    defn grad_checkpoint_gather(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.gather(x, Nx.tensor([[0], [2]])))
        end)
      end)
    end

    defn grad_gather_plain(x) do
      grad(x, fn x -> Nx.sum(Nx.gather(x, Nx.tensor([[0], [2]]))) end)
    end

    test "gather inside checkpoint" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      assert_equal(grad_checkpoint_gather(x), grad_gather_plain(x))
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
  end

  # --- An input is also the grad target ---

  describe "input is the grad target" do
    defn grad_capture_is_target(w, x) do
      grad(w, fn w ->
        checkpoint([x, w], fn x, w ->
          Nx.sum(Nx.dot(x, w))
        end)
      end)
    end

    defn grad_capture_is_target_plain(w, x) do
      grad(w, fn w -> Nx.sum(Nx.dot(x, w)) end)
    end

    test "grad target passed as a checkpoint input" do
      w = Nx.tensor([[0.5], [0.3]])
      x = Nx.tensor([1.0, 2.0])
      assert_equal(grad_capture_is_target(w, x), grad_capture_is_target_plain(w, x))
    end
  end

  # --- Named tensors ---

  describe "named tensors" do
    defn grad_checkpoint_named(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.sin(x))
        end)
      end)
    end

    test "preserves behavior with named tensors" do
      x = Nx.tensor([1.0, 2.0, 3.0], names: [:features])
      expected = Nx.Defn.grad(x, &Nx.sum(Nx.sin(&1)))
      assert_equal(grad_checkpoint_named(x), expected)
    end
  end

  # --- Asymmetric chain (different functions per checkpoint) ---

  describe "asymmetric checkpoint chain" do
    defn grad_asymmetric(x) do
      grad(x, fn x ->
        x
        |> checkpoint(fn x -> Nx.exp(x) end)
        |> checkpoint(fn x -> Nx.tanh(x) end)
        |> checkpoint(fn x -> Nx.log(Nx.abs(x) + 1.0e-8) end)
        |> Nx.sum()
      end)
    end

    defn grad_asymmetric_plain(x) do
      grad(x, fn x ->
        x |> Nx.exp() |> Nx.tanh() |> then(&Nx.log(Nx.abs(&1) + 1.0e-8)) |> Nx.sum()
      end)
    end

    test "chain of different functions" do
      x = Nx.tensor([0.1, 0.5, 1.0])
      assert_equal(grad_asymmetric(x), grad_asymmetric_plain(x))
    end
  end

  # --- LinAlg operations inside checkpoint ---

  describe "linear algebra inside checkpoint" do
    defn grad_checkpoint_linalg(x) do
      grad(x, fn x ->
        checkpoint(x, fn x ->
          Nx.sum(Nx.LinAlg.norm(x))
        end)
      end)
    end

    defn grad_linalg_plain(x) do
      grad(x, fn x -> Nx.sum(Nx.LinAlg.norm(x)) end)
    end

    test "Nx.LinAlg.norm inside checkpoint" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      assert_equal(grad_checkpoint_linalg(x), grad_linalg_plain(x))
    end
  end

  # --- Recomputation behavior ---

  describe "recomputation" do
    defn checkpoint_used_twice(x) do
      y = checkpoint(x, fn x -> Nx.sin(x) end)
      Nx.add(y, Nx.multiply(y, 2.0))
    end

    defn plain_used_twice(x) do
      y = Nx.sin(x)
      Nx.add(y, Nx.multiply(y, 2.0))
    end

    test "checkpoint output used by multiple downstream ops gives correct result" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(checkpoint_used_twice(x), plain_used_twice(x))
    end

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

    test "baseline: without checkpoint the body runs exactly once in value_and_grad" do
      parent = self()

      {_value, _grad} =
        Nx.Defn.value_and_grad(Nx.tensor([1.0, 2.0]), fn x ->
          Nx.sum(counted_body(x, parent))
        end)

      assert_received :body_ran
      refute_received :body_ran
    end

    test "value_and_grad runs the checkpointed body twice: forward + rematerialized backward" do
      parent = self()

      {_value, _grad} =
        Nx.Defn.value_and_grad(Nx.tensor([1.0, 2.0]), fn x ->
          x
          |> checkpoint(fn x -> counted_body(x, parent) end)
          |> Nx.sum()
        end)

      assert_received :body_ran
      assert_received :body_ran
      refute_received :body_ran
    end

    test "the forward pass runs the body once however many times the output is used" do
      parent = self()

      fun = fn x ->
        y = checkpoint(x, fn x -> counted_body(x, parent) end)
        Nx.add(y, Nx.multiply(y, 2.0))
      end

      Nx.Defn.jit_apply(fun, [Nx.tensor([1.0, 2.0])])

      assert_received :body_ran
      refute_received :body_ran
    end
  end
end
