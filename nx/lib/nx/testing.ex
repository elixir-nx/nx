defmodule Nx.Testing do
  @moduledoc """
  Testing functions for Nx tensor assertions.

  This module provides functions for asserting tensor equality and
  approximate equality within specified tolerances. Both helpers handle
  vectorized tensors and produce a numeric diagnostic (max absolute /
  relative difference) on failure so that bit-level disagreements
  hidden by truncated `inspect` output are still diagnosable.
  """

  import ExUnit.Assertions
  import Nx, only: [is_tensor: 1]

  @doc """
  Asserts that two tensors are exactly equal.

  This handles NaN values correctly by considering NaN == NaN as true.
  Works with vectorized tensors — two tensors must share the same
  vectorized axes to be considered equal.
  """
  def assert_equal(left, right) when not is_tensor(left) or not is_tensor(right) do
    if not Nx.Defn.Composite.compatible?(left, right, &tensor_equal?/2) do
      flunk("""
      Tensor assertion failed.
      left: #{inspect(left)}
      right: #{inspect(right)}
      """)
    end
  end

  def assert_equal(left, right) do
    if not tensor_equal?(left, right) do
      flunk("""
      Tensor assertion failed.

      left:

      #{inspect(left)}

      right:

      #{inspect(right)}

      #{diagnose_difference(left, right)}
      """)
    end
  end

  defp tensor_equal?(left, right) do
    cond do
      not is_tensor(left) or not is_tensor(right) ->
        false

      left.vectorized_axes != right.vectorized_axes ->
        false

      Nx.shape(left) != Nx.shape(right) ->
        false

      true ->
        both_nan = Nx.is_nan(left) |> Nx.logical_and(Nx.is_nan(right))

        left
        |> Nx.equal(right)
        |> Nx.logical_or(both_nan)
        |> Nx.all()
        |> Nx.to_flat_list()
        |> Enum.all?(&(&1 == 1))
    end
  end

  @doc """
  Asserts that two tensors are approximately equal within the given tolerances.

  Works with vectorized tensors — the comparison is per vectorized
  instance, then aggregated. Two tensors must share the same vectorized
  axes to be comparable.

  See also:

  * `Nx.all_close/2` - The underlying function that performs the comparison.

  ## Options

    * `:atol` - The absolute tolerance. Defaults to 1.0e-4.
    * `:rtol` - The relative tolerance. Defaults to 1.0e-4.
  """
  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    equals =
      left.vectorized_axes == right.vectorized_axes and
        Nx.shape(left) == Nx.shape(right) and
        left
        |> Nx.all_close(right, atol: atol, rtol: rtol)
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.to_flat_list()
        |> Enum.all?(&(&1 == 1))

    if !equals do
      flunk("""
      expected

      #{inspect(left)}

      to be within tolerance of

      #{inspect(right)}

      (atol: #{atol}, rtol: #{rtol})

      #{diagnose_difference(left, right)}
      """)
    end
  end

  # Produces a human-readable diagnostic describing how two tensors differ.
  # If vectorized axes or shapes don't line up, returns a structural message.
  # Otherwise computes max absolute and max relative difference across all
  # elements (including vec axes) so bit-level disagreements hidden by
  # truncated `inspect` output are still visible in the failure message.
  defp diagnose_difference(left, right) when is_tensor(left) and is_tensor(right) do
    cond do
      left.vectorized_axes != right.vectorized_axes ->
        "vectorized_axes differ: left #{inspect(left.vectorized_axes)}, " <>
          "right #{inspect(right.vectorized_axes)}"

      Nx.shape(left) != Nx.shape(right) ->
        "shapes differ: left #{inspect(Nx.shape(left))}, " <>
          "right #{inspect(Nx.shape(right))}"

      true ->
        numeric_diagnostic(left, right)
    end
  end

  defp diagnose_difference(_, _), do: ""

  defp numeric_diagnostic(left, right) do
    # Devectorize so reductions collapse across vec axes too, and so
    # `Nx.to_number` on the final scalar doesn't hit a vectorized tensor.
    left = if left.vectorized_axes == [], do: left, else: Nx.devectorize(left, keep_names: false)
    right = if right.vectorized_axes == [], do: right, else: Nx.devectorize(right, keep_names: false)

    # Promote to a common numeric type so subtraction works for int/float mixes.
    {left_f, right_f} =
      case {Nx.type(left), Nx.type(right)} do
        {{:f, _}, {:f, _}} -> {left, right}
        {{:c, _}, _} -> {left, Nx.as_type(right, Nx.type(left))}
        {_, {:c, _}} -> {Nx.as_type(left, Nx.type(right)), right}
        _ -> {Nx.as_type(left, {:f, 32}), Nx.as_type(right, {:f, 32})}
      end

    diff = Nx.subtract(left_f, right_f) |> Nx.abs()
    max_abs = diff |> Nx.reduce_max() |> Nx.to_number()

    # Relative diff: |a - b| / max(|a|, |b|, tiny) to avoid divide by zero.
    denom =
      Nx.max(Nx.abs(left_f), Nx.abs(right_f))
      |> Nx.max(Nx.tensor(1.0e-30))

    max_rel = Nx.divide(diff, denom) |> Nx.reduce_max() |> Nx.to_number()

    "max absolute difference: #{inspect(max_abs)}\n" <>
      "max relative difference: #{inspect(max_rel)}"
  rescue
    # If the diff computation itself fails (mixed complex/real, NaN propagation,
    # unusual types, etc.), fall back silently — the inspect output above is
    # still shown.
    _ -> ""
  end
end
