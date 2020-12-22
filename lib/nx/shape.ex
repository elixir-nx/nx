defmodule Nx.Shape do
  @moduledoc false

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

  The dimensions of `shape` is expanded to match the
  dimensions of `new_shape` according to the axes
  mapping.

  ## Examples

  ### Scalars

      iex> Nx.Shape.broadcast({}, {4, 2, 1, 5}, [])
      {4, 2, 1, 5}

      iex> Nx.Shape.broadcast({}, {}, [])
      {}

  ### n-D shapes

      iex> Nx.Shape.broadcast({1}, {2, 3, 4}, [2])
      {2, 3, 4}

      iex> Nx.Shape.broadcast({4, 2, 3}, {4, 3, 4, 2, 3}, [2, 3, 4])
      {4, 3, 4, 2, 3}

  ### Custom axes

      iex> Nx.Shape.broadcast({2}, {2, 3}, [0])
      {2, 3}

  ### Error cases

      iex> Nx.Shape.broadcast({4, 2, 2}, {1, 1}, [0, 1, 2])
      ** (ArgumentError) cannot broadcast tensor of dimensions {4, 2, 2} to {1, 1} with axes [0, 1, 2]

      iex> Nx.Shape.broadcast({2, 2}, {2, 2, 2}, [1, 0])
      ** (ArgumentError) broadcast axes must be ordered, got 0 after 1
  """
  def broadcast(old_shape, new_shape, axes)

  def broadcast(old_shape, new_shape, axes)
      when is_tuple(old_shape) and is_tuple(new_shape) and is_list(axes) do
    old_rank = tuple_size(old_shape)
    new_rank = tuple_size(new_shape)

    if length(axes) != old_rank do
      raise ArgumentError,
            "expected length of axes (#{length(axes)}) to match rank of shape (#{old_rank})"
    end

    if old_rank > new_rank or not valid_broadcast?(axes, 0, -1, old_shape, new_shape) do
      raise ArgumentError,
            "cannot broadcast tensor of dimensions #{inspect(old_shape)} " <>
            "to #{inspect(new_shape)} with axes #{inspect(axes)}"
    end

    new_shape
  end

  defp valid_broadcast?([head | tail], axis, last, old_shape, new_shape) do
    if head < last do
      raise ArgumentError, "broadcast axes must be ordered, got #{head} after #{last}"
    end

    old_dim = elem(old_shape, axis)
    new_dim = elem(new_shape, head)

    (old_dim == 1 or old_dim == new_dim) and
      valid_broadcast?(tail, axis + 1, head, old_shape, new_shape)
  end

  defp valid_broadcast?([], _axis, _head, _old_shape, _new_shape), do: true

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
      ** (ArgumentError) cannot broadcast tensor of dimensions {4, 2, 5} to {3, 2, 5}
  """
  def binary_broadcast(left_shape, right_shape)

  def binary_broadcast(shape, shape), do: shape

  def binary_broadcast(left_shape, right_shape)
      when is_tuple(left_shape) and is_tuple(right_shape) do
    left_rank = tuple_size(left_shape)
    right_rank = tuple_size(right_shape)
    rank = max(left_rank, right_rank)

    left_lower = Nx.Shared.shape_to_lower_ranked_list(left_shape, left_rank, rank)
    right_lower = Nx.Shared.shape_to_lower_ranked_list(right_shape, right_rank, rank)

    case binary_broadcast(left_lower, right_lower, []) do
      {:ok, new} ->
        new

      :error ->
        raise ArgumentError,
              "cannot broadcast tensor of dimensions #{inspect(left_shape)} " <>
                "to #{inspect(right_shape)}"
    end
  end

  defp binary_broadcast([ldim | ldims], [rdim | rdims], acc)
       when rdim == 1 or ldim == 1 or rdim == ldim,
       do: binary_broadcast(ldims, rdims, [max(rdim, ldim) | acc])

  defp binary_broadcast([], [], acc),
    do: {:ok, List.to_tuple(acc)}

  defp binary_broadcast(_, _, _),
    do: :error

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
      ** (ArgumentError) dot/zip expects shapes to be compatible, dimension 1 of left-side (1) does not equal dimension 0 of right-side (2)
  """
  def dot(s1, s2) do
    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} ->
        binary_broadcast(s1, s2)

      {_, 0} ->
        binary_broadcast(s1, s2)

      {n, 1} ->
        zip_reduce(s1, [n - 1], s2, [0])

      {1, m} ->
        zip_reduce(s1, [0], s2, [m - 2])

      {n, m} when n >= 2 and m >= 2 ->
        zip_reduce(s1, [n - 1], s2, [m - 2])
    end
  end

  @doc """
  Computes the shape for zip_reduce.

  In order for the dimensions to be correct, the value of each shape
  at the given axes must match. It expects axes to have already been
  normalized.

  ## Examples

      iex> Nx.Shape.zip_reduce({1, 2, 3}, [0, 1], {3, 1, 2}, [1, 2])
      {3, 3}

      iex> Nx.Shape.zip_reduce({1, 2, 3}, [0, 1], {1, 2, 3}, [1, 2])
      ** (ArgumentError) dot/zip expects shapes to be compatible, dimension 0 of left-side (1) does not equal dimension 1 of right-side (2)

  """
  def zip_reduce(s1, axes1, s2, axes2) do
    validate_zip_reduce_axes!(s1, axes1, s2, axes2)
    outer(contract(s1, axes1), contract(s2, axes2))
  end

  def validate_zip_reduce_axes!(s1, [a1 | axes1], s2, [a2 | axes2]) do
    d1 = elem(s1, a1)
    d2 = elem(s2, a2)

    if d1 == d2 do
      validate_zip_reduce_axes!(s1, axes1, s2, axes2)
    else
      raise ArgumentError,
            "dot/zip expects shapes to be compatible," <>
              " dimension #{a1} of left-side (#{d1}) does not equal" <>
              " dimension #{a2} of right-side (#{d2})"
    end
  end

  def validate_zip_reduce_axes!(_, [], _, []) do
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

      iex> Nx.Shape.transpose_axes({})
      []
      iex> Nx.Shape.transpose_axes({3, 2, 1})
      [2, 1, 0]

  """
  def transpose_axes(shape) do
    rank = rank(shape)
    count_down(rank, rank - 1)
  end

  @doc """
  Compute the broadcast axes based on the shape rank.

  It doesn't validate if the remaining dimensions are
  actually valid.

  ## Examples

      iex> Nx.Shape.broadcast_axes({2, 2, 2}, {2, 2, 2, 2})
      [1, 2, 3]

      iex> Nx.Shape.broadcast_axes({2, 2, 2}, {2, 2, 2, 2, 2})
      [2, 3, 4]

  """
  def broadcast_axes(shape, new_shape) when tuple_size(shape) > tuple_size(new_shape) do
    raise ArgumentError,
          "cannot broadcast tensor of dimensions #{inspect(shape)} " <>
            "to #{inspect(new_shape)}"
  end

  def broadcast_axes(shape, new_shape) do
    min_size = rank(shape)
    max_size = rank(new_shape)
    count_up(min_size, max_size - min_size)
  end

  @doc """
  Converts a shape to axes.

  ## Examples

      iex> Nx.Shape.to_axes({})
      []

      iex> Nx.Shape.to_axes({2, 2, 2})
      [0, 1, 2]

  """
  def to_axes(shape), do: count_up(rank(shape), 0)

  ## Helpers

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  defp count_down(0, _n), do: []
  defp count_down(i, n), do: [n | count_down(i - 1, n - 1)]

  defp to_zero(0), do: [0]
  defp to_zero(n), do: [n | to_zero(n - 1)]

  @doc """
  Output shape after a strided operation.

  Assumes stride is validated.
  """
  def stride(shape, stride) do
    List.to_tuple(Enum.reverse(strided_dims(Tuple.to_list(shape), Tuple.to_list(stride))))
  end

  defp strided_dims([], []), do: []

  defp strided_dims([dim | shape], [s | strides]),
    do: [div(dim, s) | strided_dims(shape, strides)]

  @doc """
  Validates the window size according to the shape.
  """
  def validate_window!(shape, window)

  def validate_window!(shape, window) when tuple_size(shape) != tuple_size(window),
    do: raise ArugmentError, "invalid window dimensions, rank of shape (#{tuple_size(shape)})" <>
                             " does not match rank of window (#{tuple_size(window)})"

  def validate_window!(shape, window) when is_tuple(shape) and is_tuple(window),
    do: validate_window!(Tuple.to_list(shape), Tuple.to_list(window))

  def validate_window!([], []), do: :ok

  def validate_window!([d1 | dims], [w1 | window]) when d1 >= w1 and w1 > 0,
    do: validate_window!(dims, window)

  def validate_window!([d1 | _], [w1 | _]) do
    raise ArgumentError, "invalid window dimensions, size of window dimension #{w1}" <>
                         " is invalid for dimension of shape #{d1}"
  end

  @doc """
  Validates the strides according to the shape.
  """
  def validate_strides!(shape, strides)

  def validate_strides!(shape, strides) when tuple_size(strides) != tuple_size(shape),
    do: raise ArgumentError, "invalid stride dimensions, rank of shape (#{tuple_size(shape)})" <>
                             " does not match rank of strides (#{tuple_size(strides)})"

  def validate_strides!(shape, strides) when is_tuple(shape) and is_tuple(strides),
    do: validate_strides!(Tuple.to_list(shape), Tuple.to_list(strides))

  def validate_strides!([], []), do: :ok

  def validate_strides!([d1 | dims], [s1 | strides]) when d1 >= s1 and s1 > 0,
    do: validate_strides!(dims, strides)

  def validate_strides!([d1 | _], [s1 | _]) do
    raise ArgumentError, "invalid stride dimensions, size of stride (#{s1})" <>
                         " exceeds size of shape (#{d1})" <>
                         " in dimension"
  end
end
