defmodule Nx.Testing do
  @moduledoc """
  Testing functions for Nx tensor assertions.

  This module provides functions for asserting tensor equality and
  approximate equality within specified tolerances.
  """

  import ExUnit.Assertions
  import Nx, only: [is_tensor: 1]

  @doc """
  Asserts that two tensors are exactly equal.

  This handles NaN values correctly by considering NaN == NaN as true.
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
      left: #{inspect(left)}
      right: #{inspect(right)}
      """)
    end
  end

  defp tensor_equal?(left, right) do
    both_nan = Nx.is_nan(left) |> Nx.logical_and(Nx.is_nan(right))

    left
    |> Nx.equal(right)
    |> Nx.logical_or(both_nan)
    |> Nx.all()
    |> Nx.to_flat_list()
    |> Enum.all?(&(&1 == 1))
  end

  @doc """
  Asserts that two tensors are approximately equal within the given tolerances.

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
      """)
    end
  end
end
