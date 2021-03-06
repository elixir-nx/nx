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

      iex> Index.project_on_axis(Weights.build({3, 4, 5}), 2, 1)
      1

  """
  def project_on_axis(weights, axis, i) do
    w = Weights.weight_of_axis(weights, axis)
    size = Weights.size(weights)
    rem(i * w, size)
  end

  # defp index_check!(shape, i) do
  #   if i >= tuple_size(shape) do
  #     raise ArgumentError, "index #{i} is out-of-range for shape #{inspect(shape)}"
  #   end
  # end
end