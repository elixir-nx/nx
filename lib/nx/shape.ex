defmodule Nx.Shape do
  @moduledoc false

  # TODO: Change Nx/Nx.Util module to use Nx.Shape whenever possible

  @doc """
  Computes the rank of a shape.

  ## Examples

      iex> Nx.Shape.rank({1, 2, 3})
      3

  """
  def rank(shape), do: tuple_size(shape)

  @doc """
  Computes the size of a shape.

  ## Examples

      iex> Nx.Shape.size({1, 2, 3})
      6

  """
  def size(shape), do: tuple_product(shape, tuple_size(shape))

  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

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
      ** (ArgumentError) cannot broadcast tensor of dimensions {4, 2, 3} to {3, 2, 3}
  """
  def broadcast(old_shape, new_shape)

  def broadcast(shape, shape), do: shape

  def broadcast(old_shape, new_shape) when is_tuple(old_shape) and is_tuple(new_shape) do
    old_rank = tuple_size(old_shape)
    new_rank = tuple_size(new_shape)
    rank = max(old_rank, new_rank)

    old_lower = shape_to_lower_ranked_list(old_shape, old_rank, rank)
    new_lower = shape_to_lower_ranked_list(new_shape, new_rank, rank)

    case unary_broadcast(old_lower, new_lower, []) do
      {:ok, new} ->
        List.to_tuple(new)

      :error ->
        raise ArgumentError,
              "cannot broadcast tensor of dimensions #{inspect(old_shape)} " <>
                "to #{inspect(new_shape)}"
    end
  end

  defp unary_broadcast([odim | odims], [ndim | ndims], acc)
       when ndim == 1 or ndim == odim,
       do: unary_broadcast(odims, ndims, [odim | acc])

  defp unary_broadcast([1 | odims], [ndim | ndims], acc),
    do: unary_broadcast(odims, ndims, [ndim | acc])

  defp unary_broadcast([], [], acc),
    do: {:ok, acc}

  defp unary_broadcast(_, _, _),
    do: :error

  defp shape_to_lower_ranked_list(_tuple, 0, 0),
    do: []

  defp shape_to_lower_ranked_list(tuple, 0, rank),
    do: [1 | shape_to_lower_ranked_list(tuple, 0, rank - 1)]

  defp shape_to_lower_ranked_list(tuple, size, rank),
    do: [:erlang.element(size, tuple) | shape_to_lower_ranked_list(tuple, size - 1, rank - 1)]

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

  # TODO: don't use length here because it may be expensive.
  # We can compute the rank and order by rank.
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
    raise ArgumentError,
          "could not broadcast shapes because dimensions are" <>
            " incompatible, expected dimensions to be equal or" <>
            " either dimension to be 1, got: #{dim1} and #{dim2}"
  end

  @doc """
  Contracts a shape along the given axes.

  It expects the axes to have been normalized.

  ## Examples

      iex> Nx.Shape.contract({4, 1, 2}, [1])
      {4, 2}

      iex> Nx.Shape.contract({2, 4, 6, 5}, [1, 3])
      {2, 6}

      iex> Nx.Shape.contract({1, 2, 3}, [])
      {1, 2, 3}

      iex> Nx.Shape.contract({4, 2, 8}, [2])
      {4, 2}

  """
  def contract(shape, axes) do
    List.to_tuple(contract(shape, axes, 0, tuple_size(shape)))
  end

  defp contract(_shape, _axes, n, n) do
    []
  end

  defp contract(shape, axes, i, n) do
    if i in axes do
      contract(shape, axes, i + 1, n)
    else
      [elem(shape, i) | contract(shape, axes, i + 1, n)]
    end
  end

  @doc """
  Transposes a shape according to the given permutation.

  ## Examples

    iex> Nx.Shape.transpose({4, 8, 2, 1}, [1, 0, 3, 2])
    {8, 4, 1, 2}

  ### Error cases

    iex> Nx.Shape.transpose({4, 8, 2, 1}, [0, 1, 2])
    ** (ArgumentError) expected length of permutation (3) to match rank of shape (4)

  """
  def transpose(shape, permutation)

  def transpose(shape, permutation) when tuple_size(shape) == length(permutation) do
    List.to_tuple(Enum.map(permutation, &elem(shape, &1)))
  end

  def transpose(shape, permutation) do
    raise ArgumentError,
          "expected length of permutation (#{length(permutation)})" <>
            " to match rank of shape (#{tuple_size(shape)})"
  end

  @doc """
  Computes the shape of the dot operation.

  The shape is contracted along specific dimensions or
  broadcasted according to the semantics of the dot product
  operation described in the `Nx.dot/2` documentation.

  ## Examples

  ### Scalars
      iex> Nx.Shape.dot({}, {2, 3, 2})
      {2, 3, 2}

      iex> Nx.Shape.dot({2, 1}, {})
      {2, 1}

  ### Vectors

      iex> Nx.Shape.dot({5}, {5})
      {}

  ### Matrices and n-D tensors

      iex> Nx.Shape.dot({2, 2}, {2, 3})
      {2, 3}

      iex> Nx.Shape.dot({2, 3, 2}, {3, 2, 3})
      {2, 3, 3, 3}

  ### Error cases

      iex> Nx.Shape.dot({2, 1}, {2, 2})
      ** (ArgumentError) dot product expects shapes to be compatible, dimension 1 of left-side (1) does not equal dimension 0 of right-side (2)
  """
  def dot(s1, s2) do
    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} ->
        binary_broadcast(s1, s2)

      {_, 0} ->
        binary_broadcast(s1, s2)

      {n, 1} ->
        validate_dot_axes!([n - 1], s1, [0], s2)
        dot(s1, [n - 1], s2, [0])

      {1, m} ->
        validate_dot_axes!([0], s1, [m - 2], s2)
        dot(s1, [0], s2, [m - 2])

      {n, m} when n >= 2 and m >= 2 ->
        validate_dot_axes!([n - 1], s1, [m - 2], s2)
        dot(s1, [n - 1], s2, [m - 2])
    end
  end

  defp dot(s1, axes1, s2, axes2), do: outer(contract(s1, axes1), contract(s2, axes2))

  @doc """
  Validates the contraction dimensions of a dot product are correct.

  In order for the dimensions to be correct, the value of each shape
  at the given axes must match.

  ## Examples

      iex> Nx.Shape.validate_dot_axes!([0, 1], {1, 2, 3}, [1, 2], {3, 1, 2})
      :ok

      iex> Nx.Shape.validate_dot_axes!([0, 1], {1, 2, 3}, [1, 2], {1, 2, 3})
      ** (ArgumentError) dot product expects shapes to be compatible, dimension 0 of left-side (1) does not equal dimension 1 of right-side (2)
  """
  def validate_dot_axes!([a1 | axes1], s1, [a2 | axes2], s2) do
    d1 = elem(s1, a1)
    d2 = elem(s2, a2)

    if d1 == d2 do
      validate_dot_axes!(axes1, s1, axes2, s2)
    else
      raise ArgumentError,
            "dot product expects shapes to be compatible," <>
              " dimension #{a1} of left-side (#{d1}) does not equal" <>
              " dimension #{a2} of right-side (#{d2})"
    end
  end

  def validate_dot_axes!([], _s1, [], _s2) do
    :ok
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
  def outer(s1, s2) do
    l1 = Tuple.to_list(s1)
    l2 = Tuple.to_list(s2)
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
    unless size(shape) == size(new_shape) do
      raise ArgumentError,
            "cannot reshape, current shape #{inspect(shape)} is not compatible with " <>
              "new shape #{inspect(new_shape)}"
    end

    new_shape
  end

  @doc """
  Normalize the axis to the given shape.

  ## Examples

      iex> Nx.Shape.normalize_axis({4, 2, 3}, -1)
      2

      iex> Nx.Shape.normalize_axis({4, 2, 1, 4}, -2)
      2

      iex> Nx.Shape.normalize_axis({4, 2, 1, 4}, 1)
      1

  ### Error cases

      iex> Nx.Shape.normalize_axis({4, 2, 5}, -4)
      ** (ArgumentError) given axis (-4) invalid for shape with rank 3

      iex> Nx.Shape.normalize_axis({4, 2, 5}, 3)
      ** (ArgumentError) given axis (3) invalid for shape with rank 3
  """
  def normalize_axis(shape, axis)

  def normalize_axis(shape, axis) when axis < 0 and abs(axis) <= tuple_size(shape),
    do: tuple_size(shape) + axis

  def normalize_axis(shape, axis) when axis >= 0 and axis < tuple_size(shape),
    do: axis

  def normalize_axis(shape, axis) do
    raise ArgumentError,
          "given axis (#{axis}) invalid for shape with rank #{tuple_size(shape)}"
  end

  @doc """
  Normalize a list of unique axis.

  See `normalize_axis/1`.

  ## Examples

      iex> Nx.Shape.normalize_axes({2, 3, 4}, [-1, 0])
      [2, 0]

  ### Error Cases

      iex> Nx.Shape.normalize_axes({2, 3, 4}, [1, 1])
      ** (ArgumentError) axes [1, 1] must be unique integers between 0 and 2
  """
  def normalize_axes(shape, axes) when is_list(axes) do
    normalized = Enum.map(axes, &normalize_axis(shape, &1))

    if length(Enum.uniq(normalized)) != length(axes) do
      raise ArgumentError,
            "axes #{inspect(axes)} must be unique integers between 0 and #{tuple_size(shape) - 1}"
    end

    normalized
  end

  @doc """
  Returns the axes for transposition.

  ## Examples

      iex> Nx.Shape.transpose_axes({3, 2, 1})
      [2, 1, 0]

  """
  def transpose_axes(shape), do: to_zero(rank(shape) - 1)

  defp to_zero(0), do: [0]
  defp to_zero(n), do: [n | to_zero(n - 1)]
end
