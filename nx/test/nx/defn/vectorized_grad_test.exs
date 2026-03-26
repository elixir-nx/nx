defmodule Nx.Defn.VectorizedGradTest do
  @moduledoc """
  Correctness tests for vectorized gradients.

  For each operation, computes the gradient two ways:
  1. Vectorized: grad on the full vectorized tensor
  2. Per-element: grad on each batch element separately, then stack

  If they don't match, there is a bug in the vectorized gradient machinery.
  """
  use ExUnit.Case, async: true

  import Nx.Defn

  @atol 1.0e-4

  # ── Test helper ────────────────────────────────────────────────────

  defp check_vectorized_grad(x_data, fun, opts \\ []) do
    atol = opts[:atol] || @atol
    batch_size = elem(Nx.shape(x_data), 0)
    inner_shape = x_data.shape |> Tuple.to_list() |> tl() |> List.to_tuple()

    x_vec = Nx.vectorize(x_data, :batch)
    vec_grad = Nx.Defn.grad(x_vec, fun)
    vec_grad_devec = Nx.devectorize(vec_grad, keep_names: false)

    per_element_grads =
      for i <- 0..(batch_size - 1) do
        x_i = Nx.reshape(x_data[i], inner_shape)
        Nx.Defn.grad(x_i, fun)
      end

    stacked = Nx.stack(per_element_grads)

    for i <- 0..(batch_size - 1) do
      vec_slice = vec_grad_devec[i] |> Nx.reshape(inner_shape)
      elem_slice = stacked[i] |> Nx.reshape(inner_shape)

      for {v, e} <- Enum.zip(Nx.to_flat_list(vec_slice), Nx.to_flat_list(elem_slice)) do
        if v == :nan and e == :nan do
          :ok
        else
          assert_in_delta v, e, atol, "Mismatch at batch #{i}: vec=#{v}, elem=#{e}"
        end
      end
    end

    :ok
  end

  # ── Unary ops ───────────────────────────────────────────────────────

  describe "unary ops" do
    test "exp" do
      x = Nx.tensor([[0.5, 1.0, 1.5], [2.0, 0.3, 0.8], [1.2, 0.7, 0.1]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.exp(x)) end)
    end

    test "abs" do
      x = Nx.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.abs(x)) end)
    end

    test "negate" do
      x = Nx.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.negate(x)) end)
    end

    test "sigmoid" do
      x = Nx.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.sigmoid(x)) end)
    end

    test "cbrt" do
      x = Nx.tensor([[1.0, 8.0, 27.0], [64.0, 125.0, 216.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.cbrt(x)) end)
    end

    test "expm1/log1p" do
      x = Nx.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.log1p(Nx.expm1(x))) end)
    end

    test "pow(x, 3)" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.pow(x, 3)) end)
    end

    test "clip" do
      x = Nx.tensor([[1.0, 5.0, 10.0], [0.5, 3.0, 8.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.clip(x, 2, 7)) end)
    end

    test "conjugate" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.conjugate(x)) end)
    end

    test "remainder" do
      x = Nx.tensor([[5.0, 7.0, 9.0], [11.0, 13.0, 15.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.remainder(x, 3)) end)
    end

    test "as_type" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.as_type(x, :f64)) end)
    end

    test "trig functions" do
      x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, -0.5, 0.2]], type: :f32)
      x_pos = Nx.tensor([[1.5, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)

      for {_name, fun, input} <- [
            {:acos, fn x -> Nx.sum(Nx.acos(x)) end, x},
            {:acosh, fn x -> Nx.sum(Nx.acosh(x)) end, x_pos},
            {:asin, fn x -> Nx.sum(Nx.asin(x)) end, x},
            {:asinh, fn x -> Nx.sum(Nx.asinh(x)) end, x},
            {:atan, fn x -> Nx.sum(Nx.atan(x)) end, x},
            {:atanh, fn x -> Nx.sum(Nx.atanh(x)) end, x},
            {:cos, fn x -> Nx.sum(Nx.cos(x)) end, x},
            {:cosh, fn x -> Nx.sum(Nx.cosh(x)) end, x},
            {:sinh, fn x -> Nx.sum(Nx.sinh(x)) end, x},
            {:tanh, fn x -> Nx.sum(Nx.tanh(x)) end, x},
            {:sin, fn x -> Nx.sum(Nx.sin(x)) end, x},
            {:tan, fn x -> Nx.sum(Nx.tan(x)) end, x}
          ] do
        check_vectorized_grad(input, fun)
      end
    end

    test "math functions" do
      x_pos = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, -0.5, 0.2]], type: :f32)

      for {_name, fun, input} <- [
            {:erf, fn x -> Nx.sum(Nx.erf(x)) end, x},
            {:erfc, fn x -> Nx.sum(Nx.erfc(x)) end, x},
            {:erf_inv, fn x -> Nx.sum(Nx.erf_inv(x)) end, x},
            {:rsqrt, fn x -> Nx.sum(Nx.rsqrt(x)) end, x_pos},
            {:sqrt, fn x -> Nx.sum(Nx.sqrt(x)) end, x_pos},
            {:log, fn x -> Nx.sum(Nx.log(x)) end, x_pos},
            {:exp, fn x -> Nx.sum(Nx.exp(x)) end, x}
          ] do
        check_vectorized_grad(input, fun)
      end
    end

    test "atan2" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.atan2(x, Nx.tensor(1.0))) end)
    end
  end

  # ── Binary ops ──────────────────────────────────────────────────────

  describe "binary ops" do
    test "multiply then sum" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
    end

    test "add with scalar" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.add(x, 10.0)) end)
    end

    test "multiply with broadcast" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(x, Nx.tensor([2.0, 3.0]))) end)
    end

    test "divide and subtract" do
      x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, -0.5, 0.2]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.divide(x, 2.0)) end)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.subtract(x, 0.5)) end)
    end

    test "chained binary ops" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.divide(Nx.add(Nx.multiply(x, x), x), Nx.add(x, 1)))
      end)
    end

    # Exercises broadcast_vectors alignment for mixed vectorized/non-vectorized
    test "mixed: vectorized x * non-vectorized y" do
      y = Nx.tensor([10.0, 20.0, 30.0], type: :f32)
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(x, y)) end)
    end
  end

  # ── Reductions ──────────────────────────────────────────────────────

  describe "reductions" do
    test "sum" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, &Nx.sum/1)
    end

    test "mean" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, &Nx.mean/1)
    end

    test "product" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, &Nx.product/1)
    end

    test "reduce_max" do
      x = Nx.tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]], type: :f32)
      check_vectorized_grad(x, &Nx.reduce_max/1)
    end

    test "reduce_min" do
      x = Nx.tensor([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]], type: :f32)
      check_vectorized_grad(x, &Nx.reduce_min/1)
    end

    test "composed reduction: sum(x * x)" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
    end

    # Exercises reduce_g fix for partial axis reduction with vectorized tensors
    test "sum axis 0 on 2D inner" do
      x =
        Nx.tensor(
          [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
          ],
          type: :f32
        )

      check_vectorized_grad(x, fn x -> Nx.sum(Nx.sum(x, axes: [0])) end)
    end

    test "sum with negative axis" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(x, axes: [-1]) end)
    end

    test "sum with named axis" do
      # Named axis sum works; uses inline comparison since check_vectorized_grad
      # loses dimension names when slicing per-element.
      x =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [nil, :features])
        |> Nx.vectorize(:batch)

      grad = Nx.Defn.grad(x, fn x -> Nx.sum(x, axes: [:features]) end)
      assert grad.vectorized_axes == [batch: 2]
    end

    # Exercises cumulative_sum axis name collision fix in revectorize_node
    test "cumulative_sum" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.cumulative_sum(x)) end)
    end

    test "partial axis reduction on 2D inner - axis 0" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(x, axes: [0]) |> Nx.sum() end)
    end

    test "partial axis reduction on 2D inner - axis 1" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(x, axes: [1]) |> Nx.sum() end)
    end

    test "partial axis reduction on 3D inner - non-zero axis" do
      x = Nx.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(x, axes: [1]) |> Nx.sum() end)
    end

    test "product partial axis reduction on 2D inner - axis 1" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.product(x, axes: [1]) |> Nx.sum() end)
    end

    test "reduce_min then sum" do
      x = Nx.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.reduce_min(x)) end)
    end

    test "argsort" do
      x = Nx.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(x, Nx.argsort(x))) end)
    end
  end

  # ── Shape ops ───────────────────────────────────────────────────────

  describe "shape ops" do
    test "transpose" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.transpose(x)) end)
    end

    test "squeeze" do
      x = Nx.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.squeeze(x)) end)
    end

    test "broadcast" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.broadcast(x, {3, 2})) end)
    end

    # Exercises grad(:reshape) boundary crossing between vectorized/devectorized
    test "reshape inside grad" do
      x =
        Nx.tensor(
          [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
          ],
          type: :f32
        )

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.transpose(Nx.reshape(x, {4})))
      end)
    end

    test "reshape" do
      x =
        Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], type: :f32)

      check_vectorized_grad(x, fn x -> Nx.sum(Nx.reshape(x, {2, 3})) end)
    end

    test "reverse" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.reverse(x)) end)
    end

    test "new_axis" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.new_axis(x, 0)) end)
    end

    test "tile" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.tile(x, [2])) end)
    end

    test "transpose with 2D inner shape" do
      x =
        Nx.tensor(
          [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
          ],
          type: :f32
        )

      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(Nx.transpose(x), 2.0)) end)
    end
  end

  # ── Indexing ops ────────────────────────────────────────────────────

  describe "indexing ops" do
    test "slice" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.slice(x, [1], [2])) end)
    end

    test "slice at different positions" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.slice(x, [2], [3])) end)
    end

    test "take" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.take(x, Nx.tensor([0, 2]))) end)
    end

    test "gather" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.gather(x, Nx.tensor([[0], [2]]))) end)
    end

    test "gather with multiple indices" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.gather(x, Nx.tensor([[0], [1], [3]]))) end)
    end

    test "gather with 2D inner shape" do
      x =
        Nx.tensor(
          [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
          ],
          type: :f32
        )

      check_vectorized_grad(x, fn x -> Nx.sum(Nx.gather(x, Nx.tensor([[0], [2]]))) end)
    end

    test "gather with power" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        x |> Nx.pow(2) |> Nx.gather(Nx.tensor([[0], [2]])) |> Nx.sum()
      end)
    end

    test "take_along_axis" do
      x = Nx.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], type: :f32)
      idx = Nx.tensor([2, 0, 1])
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.take_along_axis(x, idx, axis: 0)) end)
    end

    test "put_slice" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.put_slice(x, [0], Nx.tensor([99.0]))) end)
    end

    test "indexed_add" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.indexed_add(x, Nx.tensor([[0]]), Nx.tensor([10.0])))
      end)
    end

    test "indexed_put" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.indexed_put(x, Nx.tensor([[1]]), Nx.tensor([99.0])))
      end)
    end

    test "sort" do
      x = Nx.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.sort(x)) end)
    end

    test "sort with negative axis" do
      x = Nx.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.sort(x, axis: -1)) end)
    end
  end

  # ── Concatenate / stack ─────────────────────────────────────────────

  describe "concatenate and stack" do
    # Exercises concatenate grad axis offset for devectorized shapes
    test "concatenate" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.concatenate([x, Nx.multiply(x, 2)], axis: 0))
      end)
    end

    test "concatenate self" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.concatenate([x, x])) end)
    end

    test "concatenate with constant" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.concatenate([x, Nx.tensor([0.0, 0.0, 0.0])]))
      end)
    end

    test "stack" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.stack([x, Nx.multiply(x, 2)])) end)
    end
  end

  # ── Pad ─────────────────────────────────────────────────────────────

  describe "pad" do
    test "pad" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.pad(x, 0.0, [{1, 1, 0}])) end)
    end

    test "pad then sum" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.pad(x, 0.0, [{1, 1, 0}])) end)
    end
  end

  # ── Window ops ──────────────────────────────────────────────────────

  describe "window ops" do
    # Exercises window_scatter adjust_vectorized_args
    test "window_sum" do
      x =
        Nx.tensor(
          [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
            [[-1.0, 0.0, 1.0, 2.0]]
          ],
          type: :f32
        )

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.window_sum(x, {1, 2}))
      end)
    end

    test "window_sum 1D" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.window_sum(x, {2}, strides: [1])) end)
    end

    test "window_max" do
      x = Nx.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 6.0, 8.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.window_max(x, {2})) end)
    end

    test "window_min" do
      x = Nx.tensor([[4.0, 2.0, 3.0, 1.0], [8.0, 6.0, 7.0, 5.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.window_min(x, {2})) end)
    end
  end

  # ── Dot / matmul ────────────────────────────────────────────────────

  describe "dot" do
    test "dot with weight vector" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      w = Nx.tensor([0.5, -1.0, 2.0])
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.dot(x, w)) end)
    end

    test "dot then sum" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      w = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.dot(x, w)) end)
    end

    test "dot with 2D inner shape" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)
      w = Nx.tensor([0.5, -1.0])
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.dot(x, w)) end)
    end

    # Exercises captured concrete tensor handling in parents_args
    test "dot with captured matrix" do
      w = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)

      check_vectorized_grad(
        Nx.tensor(
          [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
          ],
          type: :f32
        ),
        fn x -> Nx.sum(Nx.dot(x, w)) end
      )
    end
  end

  # ── Select ──────────────────────────────────────────────────────────

  describe "select" do
    test "select (conditional)" do
      x = Nx.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.select(Nx.greater(x, 0), x, Nx.negate(x)))
      end)
    end
  end

  # ── Composed chains ─────────────────────────────────────────────────

  describe "composed chains" do
    test "transpose then squeeze then sum" do
      x = Nx.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.squeeze(Nx.transpose(x))) end)
    end

    # Exercises captured concrete tensor in Expr.parameter and parents_args
    test "captured concrete tensor in dot" do
      w = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      x = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.dot(x, w) end)
    end

    # Exercises composed op chain with captured tensors through multiple grad rules
    test "sigmoid(x @ w + b)" do
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]], type: :f32)
      b = Nx.tensor([0.1, -0.1], type: :f32)
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]], type: :f32)

      check_vectorized_grad(x, fn x ->
        Nx.sum(Nx.sigmoid(Nx.add(Nx.dot(x, w), b)))
      end)
    end

    test "captured constant" do
      w = Nx.tensor([2.0, 3.0])
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.multiply(x, w)) end)
    end

    test "captured matrix" do
      w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.dot(x, w)) end)
    end

    test "multiple same-axis inputs, grad wrt one" do
      # Both x and w share axis :a; can't use check_vectorized_grad since
      # it only handles single-input vectorization.
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> Nx.vectorize(:a)
      w = Nx.tensor([[0.5, 0.3], [0.2, 0.4]]) |> Nx.vectorize(:a)
      grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.multiply(x, w)) end)
      assert grad.vectorized_axes == [a: 2]
      expected = Nx.tensor([[0.5, 0.3], [0.2, 0.4]]) |> Nx.vectorize(:a)
      assert Nx.devectorize(grad) == Nx.devectorize(expected)
    end
  end

  # ── FFT ─────────────────────────────────────────────────────────────

  describe "fft" do
    # FFT/IFFT produce complex outputs; check_vectorized_grad's assert_in_delta
    # can't handle complex numbers, so we use shape-only assertions.
    test "fft" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]) |> Nx.vectorize(:batch)
      grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.real(Nx.fft(x))) end)
      assert grad.vectorized_axes == [batch: 2]
    end

    test "ifft" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]) |> Nx.vectorize(:batch)
      grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.real(Nx.ifft(Nx.as_type(x, :c64)))) end)
      assert grad.vectorized_axes == [batch: 2]
    end
  end

  # ── Conv ────────────────────────────────────────────────────────────

  describe "conv" do
    # Exercises grad_conv devectorize fix: the conv collapse/uncollapse chain
    # bakes the vectorized dim into batch_group_size
    test "conv with vectorized" do
      k = Nx.tensor([[[1.0, 0.0, -1.0]]])
      x = Nx.tensor([[[[1.0, 2.0, 3.0, 4.0]]], [[[5.0, 6.0, 7.0, 8.0]]]], type: :f32)
      check_vectorized_grad(x, fn x -> Nx.sum(Nx.conv(x, k)) end)
    end
  end

  # ── Linalg ──────────────────────────────────────────────────────────

  describe "linalg" do
    # Exercises Cholesky batch_transpose and matmul helpers
    test "cholesky grad" do
      x =
        Nx.tensor(
          [
            [[4.0, 2.0], [2.0, 5.0]],
            [[9.0, 3.0], [3.0, 5.0]],
            [[16.0, 4.0], [4.0, 8.0]]
          ],
          type: :f32
        )

      check_vectorized_grad(x, fn x ->
        l = Nx.LinAlg.cholesky(x)
        Nx.sum(l)
      end)
    end

    # Exercises triangular_solve batched_dot helper
    test "triangular_solve grad with captured a" do
      a = Nx.tensor([[1.0, 0.0], [2.0, 3.0]], type: :f32)

      check_vectorized_grad(
        Nx.tensor([[4.0, 5.0], [2.0, 3.0], [1.0, 1.0]], type: :f32),
        fn b -> Nx.sum(Nx.LinAlg.triangular_solve(a, b)) end
      )
    end

    test "triangular_solve with vectorized a and b" do
      # Both a and b are vectorized; can't use check_vectorized_grad since
      # the helper only handles single-input vectorization.
      a = Nx.tensor([[[1.0, 0.0], [2.0, 3.0]], [[1.0, 0.0], [1.0, 1.0]]]) |> Nx.vectorize(:batch)
      b = Nx.tensor([[4.0, 5.0], [2.0, 3.0]]) |> Nx.vectorize(:batch)
      grad = Nx.Defn.grad(b, fn b -> Nx.sum(Nx.LinAlg.triangular_solve(a, b)) end)
      assert grad.vectorized_axes == [batch: 2]
    end

    test "qr grad" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)

      check_vectorized_grad(x, fn x ->
        {q, r} = Nx.LinAlg.qr(x)
        Nx.sum(Nx.dot(q, r))
      end)
    end
  end

  # ── While / cond ────────────────────────────────────────────────────

  describe "control flow" do
    defn while_square_n(x) do
      {_i, result} =
        while {i = 0, x}, Nx.less(i, 3) do
          {i + 1, Nx.multiply(x, x)}
        end

      Nx.sum(result)
    end

    # Exercises update_grads(:while) devectorize at boundary (approach C)
    test "while loop: repeated squaring" do
      x = Nx.tensor([[0.5, 0.8], [1.2, 0.3], [0.9, 0.4]], type: :f32)
      check_vectorized_grad(x, &while_square_n/1)
    end

    defn cond_grad_fn(x) do
      s = Nx.sum(x)

      if Nx.greater(s, 0) do
        Nx.multiply(s, s)
      else
        Nx.negate(s)
      end
    end

    # Exercises cond clean_grads fix: prevents cross-contamination between
    # cond_then/cond_else nodes in vectorized cond
    test "cond with vectorized input" do
      x = Nx.tensor([[2.0, 3.0], [-5.0, -6.0], [1.0, 1.0]], type: :f32)
      check_vectorized_grad(x, &cond_grad_fn/1)
    end
  end

  # ── Multi-axis vectorization ────────────────────────────────────────

  describe "multi-axis vectorization" do
    test "sum" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)

      x_vec = x |> Nx.vectorize(:batch) |> Nx.vectorize(:seq)
      vec_grad = Nx.Defn.grad(x_vec, &Nx.sum/1)
      vec_devec = Nx.devectorize(vec_grad, keep_names: false)

      for i <- 0..1, j <- 0..1 do
        x_ij = x[i][j] |> Nx.reshape({2})
        elem_grad = Nx.Defn.grad(x_ij, &Nx.sum/1)

        for {v, e} <- Enum.zip(Nx.to_flat_list(vec_devec[i][j]), Nx.to_flat_list(elem_grad)) do
          assert_in_delta v, e, @atol
        end
      end
    end

    test "product" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]], type: :f32)

      x_vec = x |> Nx.vectorize(:batch) |> Nx.vectorize(:seq)
      vec_grad = Nx.Defn.grad(x_vec, &Nx.product/1)
      vec_devec = Nx.devectorize(vec_grad, keep_names: false)

      for i <- 0..1, j <- 0..1 do
        x_ij = x[i][j] |> Nx.reshape({2})
        elem_grad = Nx.Defn.grad(x_ij, &Nx.product/1)

        for {v, e} <- Enum.zip(Nx.to_flat_list(vec_devec[i][j]), Nx.to_flat_list(elem_grad)) do
          assert_in_delta v, e, @atol
        end
      end
    end

    # Exercises unbroadcast fix for multiple different vectorized axes
    test "two vectorized axes" do
      x = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: :f32)

      x_vec = x |> Nx.vectorize(:a) |> Nx.vectorize(:b)
      vec_grad = Nx.Defn.grad(x_vec, fn x -> Nx.sum(Nx.multiply(x, x)) end)
      vec_devec = Nx.devectorize(vec_grad, keep_names: false)

      for i <- 0..1, j <- 0..1 do
        x_ij = x[i][j] |> Nx.reshape({2})
        elem_grad = Nx.Defn.grad(x_ij, fn x -> Nx.sum(Nx.multiply(x, x)) end)

        for {v, e} <-
              Enum.zip(Nx.to_flat_list(vec_devec[i][j]), Nx.to_flat_list(elem_grad)) do
          assert_in_delta v, e, @atol
        end
      end
    end
  end
end
