defmodule Nx.FuzzClipGradTest do
  @moduledoc """
  Pin tests for incorrect gradients when `Nx.clip` participates in
  composite expressions.

  Found in the wild (exphil, 2026-07-14): adding a `clip(logits, -60, 60)`
  guard inside a stable-form binary cross-entropy silently produced
  garbage gradients — wrong signs and magnitudes even for values well
  INSIDE the clip range — killing a training run from step 0 (loss climbed
  from init, then params saturated at the clamp rails where grads are
  legitimately zero). The forward pass is correct throughout; only the
  backward is wrong. Identical wrong values on BinaryBackend/evaluator and
  EXLA, so the defect is in the `Nx.Defn.Grad` rule composition, not in
  any compiler.

  Workaround: `Nx.min(Nx.max(x, lo), hi)` — gradients verified correct
  for the same compositions.

  Cases are ordered from bare clip to the full BCE shape to bisect which
  composition breaks.
  """
  use ExUnit.Case, async: true

  import Nx.Testing

  @lo -60.0
  @hi 60.0

  # Mixed in-range and out-of-range points; targets exercise both BCE arms
  defp logits,
    do: Nx.tensor([-2.0, 0.5, 3.0, -70.0, 65.0, 1.0, -1.0, 0.25], type: :f32)

  defp targets,
    do: Nx.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], type: :f32)

  defp inside_mask,
    do: Nx.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], type: :f32)

  defp clamped(x), do: Nx.clip(x, @lo, @hi)
  defp clamped_mm(x), do: Nx.min(Nx.max(x, @lo), @hi)

  defp sigmoid(x), do: Nx.divide(1.0, Nx.add(1.0, Nx.exp(Nx.negate(x))))

  # Stable-form BCE: max(x, 0) - x*t + log(1 + exp(-|x|))
  defp stable_bce(x, t) do
    x
    |> Nx.max(0)
    |> Nx.subtract(Nx.multiply(x, t))
    |> Nx.add(Nx.log(Nx.add(1.0, Nx.exp(Nx.negate(Nx.abs(x))))))
    |> Nx.mean()
  end

  # Analytic d/dx mean(stable_bce(clamp(x), t)) = (sigmoid(clamp(x)) - t)/n,
  # masked to zero outside the clamp
  defp expected_bce_grad do
    n = Nx.size(logits())

    clamped(logits())
    |> sigmoid()
    |> Nx.subtract(targets())
    |> Nx.multiply(inside_mask())
    |> Nx.divide(n)
  end

  describe "bare and simple clip compositions (control group)" do
    test "grad of mean(clip(x)) is 1/n inside, 0 outside" do
      n = Nx.size(logits())
      grad = Nx.Defn.grad(logits(), fn x -> Nx.mean(clamped(x)) end)
      expected = Nx.divide(inside_mask(), n)
      assert_all_close(grad, expected, atol: 1.0e-6)
    end

    test "grad of mean(clip(x) * t) is t/n inside, 0 outside" do
      n = Nx.size(logits())
      t = targets()
      grad = Nx.Defn.grad(logits(), fn x -> Nx.mean(Nx.multiply(clamped(x), t)) end)
      expected = t |> Nx.multiply(inside_mask()) |> Nx.divide(n)
      assert_all_close(grad, expected, atol: 1.0e-6)
    end

    test "grad of mean(max(clip(x), 0)) matches step function inside" do
      n = Nx.size(logits())
      grad = Nx.Defn.grad(logits(), fn x -> Nx.mean(Nx.max(clamped(x), 0)) end)

      step = Nx.greater(clamped(logits()), 0) |> Nx.as_type(:f32)
      expected = step |> Nx.multiply(inside_mask()) |> Nx.divide(n)
      assert_all_close(grad, expected, atol: 1.0e-6)
    end

    test "grad of mean(log1p(exp(-|clip(x)|)))" do
      n = Nx.size(logits())

      grad =
        Nx.Defn.grad(logits(), fn x ->
          c = clamped(x)
          Nx.mean(Nx.log(Nx.add(1.0, Nx.exp(Nx.negate(Nx.abs(c))))))
        end)

      c = clamped(logits())
      # d/dc log(1+exp(-|c|)) = -sign(c) * exp(-|c|)/(1+exp(-|c|))
      e = Nx.exp(Nx.negate(Nx.abs(c)))

      expected =
        Nx.negate(Nx.sign(c))
        |> Nx.multiply(Nx.divide(e, Nx.add(1.0, e)))
        |> Nx.multiply(inside_mask())
        |> Nx.divide(n)

      assert_all_close(grad, expected, atol: 1.0e-6)
    end
  end

  describe "clip inside stable-form BCE (the wild bug)" do
    test "grad of stable_bce(clip(x), t) matches (sigmoid - t)/n masked" do
      grad = Nx.Defn.grad(logits(), fn x -> stable_bce(clamped(x), targets()) end)
      assert_all_close(grad, expected_bce_grad(), atol: 1.0e-5)
    end

    test "WORKAROUND: min∘max clamp in the same expression is correct" do
      grad = Nx.Defn.grad(logits(), fn x -> stable_bce(clamped_mm(x), targets()) end)
      assert_all_close(grad, expected_bce_grad(), atol: 1.0e-5)
    end
  end
end
