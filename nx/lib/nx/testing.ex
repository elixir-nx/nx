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
      """)
    end
  end

  defp tensor_equal?(left, right) do
    left = to_tensor(left)
    right = to_tensor(right)

    if left.vectorized_axes != right.vectorized_axes do
      flunk_vectorized_axes_mismatch(left, right)
    end

    if shapes_incompatible?(left, right) do
      flunk_shapes_mismatch(left, right)
    end

    both_nan = Nx.is_nan(left) |> Nx.logical_and(Nx.is_nan(right))

    left
    |> Nx.equal(right)
    |> Nx.logical_or(both_nan)
    |> Nx.all()
    |> Nx.to_flat_list()
    |> Enum.all?(&(&1 == 1))
  end

  # Wrap raw scalars/lists in tensors so the struct-field accesses
  # (`.vectorized_axes`) and `Nx.shape/1` below don't crash. Tensors
  # pass through unchanged.
  defp to_tensor(%Nx.Tensor{} = t), do: t
  defp to_tensor(other), do: Nx.tensor(other)

  defp flunk_vectorized_axes_mismatch(left, right) do
    flunk("""
    Vectorized axes mismatch

    left axes:  #{inspect(left.vectorized_axes)}
    right axes: #{inspect(right.vectorized_axes)}
    """)
  end

  defp flunk_shapes_mismatch(left, right) do
    flunk("""
    Shape mismatch

    left shape:  #{inspect(Nx.shape(left))}
    right shape: #{inspect(Nx.shape(right))}
    """)
  end

  defp flunk_not_close(left, right, atol, rtol) do
    abs_diff = left |> Nx.subtract(right) |> Nx.abs()
    max_abs = abs_diff |> Nx.devectorize() |> Nx.reduce_max() |> Nx.to_number()

    right_abs = Nx.abs(right)

    max_rel =
      abs_diff
      |> Nx.divide(Nx.max(right_abs, Nx.tensor(1.0e-10)))
      |> Nx.devectorize()
      |> Nx.reduce_max()
      |> Nx.to_number()

    flunk("""
    Tensors differ by more than the tolerance

    atol: #{atol}, rtol: #{rtol}
    max absolute difference: #{max_abs}
    max relative difference: #{max_rel}

    left:

    #{inspect(left)}

    right:

    #{inspect(right)}
    """)
  end

  # Genuine shape mismatches are rejected, but we still allow a scalar
  # (shape `{}`) to compare against a tensor of any shape — that's the
  # intentional "assert every element equals this scalar" pattern, and
  # rejecting it would break a large number of existing tests that
  # relied on `Nx.equal`'s broadcasting.
  defp shapes_incompatible?(left, right) do
    ls = Nx.shape(left)
    rs = Nx.shape(right)
    ls != rs and ls != {} and rs != {}
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

    left_t = to_tensor(left)
    right_t = to_tensor(right)

    if left_t.vectorized_axes != right_t.vectorized_axes do
      flunk_vectorized_axes_mismatch(left_t, right_t)
    end

    if shapes_incompatible?(left_t, right_t) do
      flunk_shapes_mismatch(left_t, right_t)
    end

    equals =
      left_t
      |> Nx.all_close(right_t, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> Enum.all?(&(&1 == 1))

    if not equals do
      flunk_not_close(left_t, right_t, atol, rtol)
    end
  end
end
