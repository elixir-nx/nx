defmodule Nx.BinaryBackend.Index do
  @moduledoc """
  Helper functions for working with zero-based indices.
  """

  alias Nx.BinaryBackend.Weights

  @doc """
  Turns coordinates into an index according to the given shape.

  ### Examples

      iex> Index.coords_to_i({3, 3, 3}, {0, 0, 0})
      0

      iex> Index.coords_to_i({3, 3, 3}, {2, 2, 2})
      26

      iex> Index.coords_to_i({3, 3, 3}, {1, 0, 0})
      9
  """
  def coords_to_i({}, {}) do
    0
  end

  def coords_to_i(shape, coords) when tuple_size(shape) == tuple_size(coords) do
    last_i = tuple_size(shape) - 1
    {size, total} = fetch_dim_and_coord!(shape, coords, last_i)
    coords_to_i(shape, coords, size, total, last_i - 1)
  end

  defp coords_to_i(_shape, _coords, _size, total, i) when i < 0 do
    total
  end

  defp coords_to_i(shape, coords, size, total, i) do
    {dim, coord} = fetch_dim_and_coord!(shape, coords, i)
    size2 = size * dim
    total2 = total + size * coord
    coords_to_i(shape, coords, size2, total2, i - 1)
  end

  defp fetch_dim_and_coord!(shape, coords, i) do
    dim = elem(shape, i)
    coord = elem(coords, i)
    if coord >= dim do
      raise ArgumentError, "at index #{i} coords #{inspect(coords)}" <> 
            " were invalid for shape #{inspect(shape)}"
    end
    {dim, coord}
  end

  @doc """
  Projects an index onto the given axis.

  Note: The index is wrapped on the given axis.

  ### Examples

      iex> Index.contract_axis(Weights.build({3, 4, 5}), 2, 1)
      1

  """
  def contract_axis(weights, axis, i) do
    w = Weights.weight_of_axis(weights, axis)
    size = Weights.size(weights)
    rem(i * w, size)
  end

  # def contract_axis_mapper(weights, axis) do
  #   w = Weights.weight_of_axis(weights, axis)
  #   size = Weights.size(weights)
    
  #   fn i ->
  #     absolute = i * w
  #     relative = rem(absolute, size)
  #     {relative, absolute}
  #   end
  # end



  # defp groups_to_aggregate_axes(shape, weights, groups) do
  #   Enum.reduce(groups, {[], []}, fn
  #     n..n ->
  #   end)
  # end

  @doc """
  For a count returns a value that can be enumerated for indices.

  Raises for negative values.

  ## Example

      iex> Index.range(0)
      []

      iex> Index.range(1)
      0..0

      iex> Index.range(10)
      0..9
  """
  def range(0), do: []
  def range(n) when n > 0, do: 0..(n - 1)

  def range_shift(start1..stop1, start2..stop2) do
    (start1 + start2)..(stop1 + stop2)
  end

  def range_split_right(start..stop, n) when n in start..stop do
    {range_forward(start..(n - 1)), n..stop}
  end

  def range_split_left(start..stop, n) when n in start..stop do
    {start..n, range_forward((n + 1)..stop)}
  end

  def range_non_negative(start.._) when start < 0 do
    []
  end

  def range_non_negative(_..stop) when stop < 0 do
    []
  end

  def range_non_negative(r) do
    r
  end

  def range_forward(start..stop) when start > stop do
    []
  end

  def range_forward(range) do
    range
  end

  def range_pop_left(start..stop) when start <= stop do
    r = (start + 1)..stop
    {start, range_forward(r)}
  end

  # defp index_check!(shape, i) do
  #   if i >= tuple_size(shape) do
  #     raise ArgumentError, "index #{i} is out-of-range for shape #{inspect(shape)}"
  #   end
  # end
end