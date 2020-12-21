defmodule Nx.Shape do
  @moduledoc """
  Primitives for manipulating shapes.
  """
  import Nx.Shared

  @doc """
  Broadcasts a shape to a new shape.

  Only the dimensions of `shape` can be expanded to match
  the dimensions of `new_shape`.

  The semantics of broadcasting match [Numpy's](https://numpy.org/doc/stable/user/basics.broadcasting.html)

  ## Examples

  ### Scalars

      iex> Nx.Shape.broadcast({}, {4, 2, 1, 5})
      {4, 2, 1, 5}

      iex> Nx.Shape.broadcast({}, {})
      {}

  ### n-D shapes

      iex> Nx.Shape.broadcast({1}, {2, 3, 4})
      {2, 3, 4}

      iex> Nx.Shape.broadcast({4, 2, 3}, {4, 3, 4, 2, 3})
      {4, 3, 4, 2, 3}

      iex> Nx.Shape.broadcast({1, 4, 2, 1}, {8, 4, 2, 10})
      {8, 4, 2, 10}

  ### Error cases

      iex> Nx.Shape.broadcast({4, 2, 3}, {3, 2, 3})
      ** (ArgumentError) could not broadcast shape to new shape because dimensions are incompatible, expected dimensions to be equal or shape's dimension to be 1, got: 4 and 3
  """
  def broadcast(shape, new_shape)

  def broadcast(shape, shape), do: shape

  def broadcast(shape, new_shape) when is_tuple(shape) and is_tuple(new_shape),
    do: List.to_tuple(do_broadcast(Tuple.to_list(shape), Tuple.to_list(new_shape)))

  defp do_broadcast([], new_shape), do: new_shape

  defp do_broadcast(shape, new_shape) when length(new_shape) > length(shape) do
    [dim2 | new_shape] = new_shape
    [dim2 | do_broadcast(shape, new_shape)]
  end

  defp do_broadcast([1 | shape], [dim2 | new_shape]) do
    [dim2 | do_broadcast(shape, new_shape)]
  end

  defp do_broadcast([dim2 | shape], [dim2 | new_shape]) do
    [dim2 | do_broadcast(shape, new_shape)]
  end

  defp do_broadcast([dim1 | shape], [dim2 | shape]) do
    raise ArgumentError, "could not broadcast shape to new shape because" <>
                         " dimensions are incompatible, expected dimensions" <>
                         " to be equal or shape's dimension to be 1, got:" <>
                         " #{dim1} and #{dim2}"
  end

  @doc """
  Broadcasts two shapes to a common shape.

  The dimensions of either shape can be expanded to match
  the dimension of the other. This differs from a normal
  broadcast, where one shapes dimensions remain fixed,
  while the other's are expanded to match.

  The semantics of broadcasting match [Numpy's](https://numpy.org/doc/stable/user/basics.broadcasting.html).

  ## Examples

  ### Scalar Shapes

      iex> Nx.Shape.binary_broadcast({}, {})
      {}
      iex> Nx.Shape.binary_broadcast({}, {4, 2, 1, 5})
      {4, 2, 1, 5}

  ### n-D Shapes

      iex> Nx.Shape.binary_broadcast({8, 1, 6, 1}, {7, 1, 5})
      {8, 7, 6, 5}
      iex> Nx.Shape.binary_broadcast({7, 1, 5}, {8, 1, 6, 1})
      {8, 7, 6, 5}
      iex> Nx.Shape.binary_broadcast({5, 4}, {1})
      {5, 4}
      iex> Nx.Shape.binary_broadcast({3, 1}, {15, 3, 5})
      {15, 3, 5}

  ### Error cases

      iex> Nx.Shape.binary_broadcast({4, 2, 5}, {3, 2, 5})
      ** (ArgumentError) could not broadcast shapes because dimensions are incompatible, expected dimensions to be equal or either dimension to be 1, got: 4 and 3
  """
  def binary_broadcast(s1, s2)

  def binary_broadcast(shape, shape), do: shape

  def binary_broadcast(s1, s2) when is_tuple(s1) and is_tuple(s2),
    do: List.to_tuple(do_binary_broadcast(Tuple.to_list(s1), Tuple.to_list(s2)))

  defp do_binary_broadcast(s1, s2) when length(s1) > length(s2) do
    [dim | s1] = s1
    [dim | do_binary_broadcast(s1, s2)]
  end

  defp do_binary_broadcast(s1, s2) when length(s2) > length(s1) do
    [dim | s2] = s2
    [dim | do_binary_broadcast(s1, s2)]
  end

  defp do_binary_broadcast([], s2), do: s2
  defp do_binary_broadcast(s1, []), do: s1
  defp do_binary_broadcast([1 | s1], [dim2 | s2]) do
    [dim2 | do_binary_broadcast(s1, s2)]
  end
  defp do_binary_broadcast([dim1 | s1], [1 | s2]) do
    [dim1 | do_binary_broadcast(s1, s2)]
  end
  defp do_binary_broadcast([dim | s1], [dim | s2]) do
    [dim | do_binary_broadcast(s1, s2)]
  end
  defp do_binary_broadcast([dim1 | _s1], [dim2 | _s2]) do
    raise ArgumentError, "could not broadcast shapes because dimensions are" <>
                         " incompatible, expected dimensions to be equal or" <>
                         " either dimension to be 1, got: #{dim1} and #{dim2}"
  end

  @doc """
  Contracts a shape along the given axis/axes.

  ## Examples

      iex> Nx.Shape.contract({4, 1, 2}, [1])
      {4, 2}

      iex> Nx.Shape.contract({2, 4, 6, 5}, [1, 3])
      {2, 6}

      iex> Nx.Shape.contract({1, 2, 3}, [])
      {1, 2, 3}

      iex> Nx.Shape.contract({4, 2, 8}, 2)
      {4, 2}

  ### Error Cases

      iex> Nx.Shape.contract({2}, [1, 2])
      ** (ArgumentError) length of axes (2) greater than rank of shape (1)
  """
  def contract(shape, axes)

  def contract(shape, axes) when tuple_size(shape) < length(axes) do
    raise ArgumentError, "length of axes (#{length(axes)}) greater" <>
                         " than rank of shape (#{tuple_size(shape)})"
  end

  def contract(shape, []), do: shape

  def contract(shape, [axis | []]), do: Tuple.delete_at(shape, axis)

  def contract(shape, axes) when is_list(axes) do
    {shape, _} =
      shape
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.filter(fn {_, i} -> i not in axes end)
      |> Enum.unzip()
    List.to_tuple(shape)
  end

  def contract(shape, axis) when is_integer(axis), do: Tuple.delete_at(shape, axis)

  @doc """
  Transposes a shape according to the given permutation.

  If no permutation is given, the permutation reverses
  the order of the axes of the given shape.

  ## Examples

    iex> Nx.Shape.transpose({1, 2, 3})
    {3, 2, 1}

    iex> Nx.Shape.transpose({4, 8, 2, 1}, [1, 0, 3, 2])
    {8, 4, 1, 2}

  ### Error cases

    iex> Nx.Shape.transpose({4, 8, 2, 1}, [0, 1, 2])
    ** (ArgumentError) expected length of permutation (3) to match rank of shape (4)
  """
  def transpose(shape, permutation \\ [])

  def transpose(shape, []) do
    shape
    |> Tuple.to_list()
    |> Enum.reverse()
    |> List.to_tuple()
  end

  def transpose(shape, permutation) when tuple_size(shape) == length(permutation) do
    {shape, _} =
      shape
      |> Tuple.to_list()
      |> Enum.zip(permutation)
      |> Enum.sort_by(fn {_, i} -> i end)
      |> Enum.unzip()
    List.to_tuple(shape)
  end

  def transpose(shape, permutation) do
    raise ArgumentError, "expected length of permutation (#{length(permutation)})" <>
                         " to match rank of shape (#{tuple_size(shape)})"
  end

  @doc """
  Returns the outer product of two shapes.

  ## Examples

      iex> Nx.Shape.outer({2, 3}, {1, 2})
      {2, 3, 1, 2}

      iex> Nx.Shape.outer({1}, {3, 2})
      {1, 3, 2}

      iex> Nx.Shape.outer({}, {})
      {}
  """
  def outer(s1, s2), do: combine_tuples(s1, s2)

  defp combine_tuples(tup1, tup2) do
    l1 = Tuple.to_list(tup1)
    l2 = Tuple.to_list(tup2)
    List.to_tuple(l1 ++ l2)
  end

  @doc """
  Reshapes a given shape to a new shape.

  ## Examples

      iex> Nx.Shape.reshape({2, 4}, {2, 2, 2})
      {2, 2, 2}

      iex> Nx.Shape.reshape({1, 2, 3}, {6})
      {6}

      iex> Nx.Shape.reshape({2, 1}, {1, 1, 1, 2})
      {1, 1, 1, 2}

  ### Error cases

      iex> Nx.Shape.reshape({4, 2}, {2, 3, 2})
      ** (ArgumentError) cannot reshape, current shape {4, 2} is not compatible with new shape {2, 3, 2}
  """
  def reshape(shape, new_shape) do
    unless tuple_product(shape) == tuple_product(new_shape),
      do: raise ArgumentError,
            "cannot reshape, current shape #{inspect(shape)} is not compatible with " <>
              "new shape #{inspect(new_shape)}"
    new_shape
  end

end