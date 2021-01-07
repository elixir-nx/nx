defmodule Nx.Shape do
  @moduledoc false

  @doc """
  Broadcasts a shape to a new shape.

  The dimensions of `shape` is expanded to match the
  dimensions of `new_shape` according to the axes
  mapping.

  ## Examples

  ### Scalars

      iex> Nx.Shape.broadcast!({}, {4, 2, 1, 5}, [])
      :ok

      iex> Nx.Shape.broadcast!({}, {}, [])
      :ok

  ### n-D shapes

      iex> Nx.Shape.broadcast!({1}, {2, 3, 4}, [2])
      :ok

      iex> Nx.Shape.broadcast!({4, 2, 3}, {4, 3, 4, 2, 3}, [2, 3, 4])
      :ok

  ### Custom axes

      iex> Nx.Shape.broadcast!({2}, {2, 3}, [0])
      :ok

  ### Error cases

      iex> Nx.Shape.broadcast!({4, 2, 2}, {1, 1}, [0, 1, 2])
      ** (ArgumentError) cannot broadcast tensor of dimensions {4, 2, 2} to {1, 1} with axes [0, 1, 2]

      iex> Nx.Shape.broadcast!({2, 2}, {2, 2, 2}, [1, 0])
      ** (ArgumentError) broadcast axes must be ordered, got 0 after 1
  """
  def broadcast!(old_shape, new_shape, axes)

  def broadcast!(old_shape, new_shape, axes)
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

    :ok
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

    left_lower = shape_to_lower_ranked_list(left_shape, left_rank, rank)
    right_lower = shape_to_lower_ranked_list(right_shape, right_rank, rank)

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

  defp shape_to_lower_ranked_list(_tuple, 0, 0),
    do: []

  defp shape_to_lower_ranked_list(tuple, 0, rank),
    do: [1 | shape_to_lower_ranked_list(tuple, 0, rank - 1)]

  defp shape_to_lower_ranked_list(tuple, size, rank),
    do: [:erlang.element(size, tuple) | shape_to_lower_ranked_list(tuple, size - 1, rank - 1)]

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
    l1 = contract(s1, axes1, 0, tuple_size(s1))
    l2 = contract(s2, axes2, 0, tuple_size(s2))
    List.to_tuple(l1 ++ l2)
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
  Calculates the padding needed for same padding accounting for stride.

  ## Examples

      iex> Nx.Shape.calculate_padding({4, 4}, {2, 2}, {1, 1})
      [{0, 1}, {0, 1}]

      iex> Nx.Shape.calculate_padding({3, 3}, {2, 2}, {2, 2})
      [{0, 1}, {0, 1}]
  """
  def calculate_padding(shape, window, strides)
      when is_tuple(shape) and is_tuple(window) and is_tuple(strides) do
    validate_window!(shape, window)
    validate_strides!(shape, strides)
    calculate_padding(Tuple.to_list(shape), Tuple.to_list(window), Tuple.to_list(strides))
  end

  def calculate_padding([], [], []), do: []

  def calculate_padding([dim | shape], [w | window], [s | strides]) do
    output_dim = ceil(dim / s)
    padding_size = max((output_dim - 1) * s + w - dim, 0)
    lo = floor(padding_size / 2)
    hi = ceil(padding_size / 2)
    [{lo, hi} | calculate_padding(shape, window, strides)]
  end

  @doc """
  Calculates the padding needed for same padding not accounting for stride.
  """
  def calculate_padding(shape, window) when is_tuple(window) and is_tuple(shape) do
    validate_window!(shape, window)
    calculate_padding(Tuple.to_list(shape), Tuple.to_list(window))
  end

  def calculate_padding([], []), do: []

  def calculate_padding([dim | shape], [w | window]) do
    padding_size = max(dim - 1 + w - dim, 0)
    lo = floor(padding_size / 2)
    hi = ceil(padding_size / 2)
    [{lo, hi} | calculate_padding(shape, window)]
  end

  @doc """
  Output shape after a convolution, already padded.
  """
  def conv(input_shape, kernel_shape, strides, padding) do
    filter_shape =
      kernel_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    spatial_dims =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    num_filters = elem(kernel_shape, 0)
    batch_size = elem(input_shape, 0)

    padded_shape =
      case padding do
        :valid ->
          input_shape

        :same ->
          padding_config = Nx.Shape.calculate_padding(spatial_dims, filter_shape)
          padding_config = [{0, 0} | padding_config]
          Nx.Shape.pad(input_shape, padding_config)

        padding_config when is_list(padding_config) ->
          padding_config = [{0, 0} | padding_config]
          Nx.Shape.pad(input_shape, padding_config)
      end

    old_spatial_dims =
      padded_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)
      |> Tuple.to_list()

    spatial_dims =
      Enum.reverse(
        do_spatial_dims(old_spatial_dims, Tuple.to_list(filter_shape), Tuple.to_list(strides))
      )

    List.to_tuple([batch_size, num_filters | spatial_dims])
  end

  defp do_spatial_dims([], [], []), do: []

  defp do_spatial_dims([cur | spatial], [f | filters], [s | strides]),
    do: [floor((cur - f) / s) + 1 | do_spatial_dims(spatial, filters, strides)]

  @doc """
  Output shape after a window operation.

  ## Examples

      iex> Nx.Shape.window({3, 3}, {2, 2}, {1, 1})
      {2, 2}

      iex> Nx.Shape.window({1, 2, 3}, {2, 1, 1}, {1, 1, 1})
      {1, 2, 3}

  ### Error cases

      iex> Nx.Shape.window({1, 2, 3}, {2, 1}, {1, 1, 1})
      ** (ArgumentError) invalid window dimensions, rank of shape (3) does not match rank of window (2)

      iex> Nx.Shape.window({1, 2, 3}, {2, 1, 1}, {1, 1})
      ** (ArgumentError) invalid stride dimensions, rank of shape (3) does not match rank of stride (2)
  """
  def window(shape, window, strides) do
    validate_window!(shape, window)
    validate_strides!(shape, strides)

    List.to_tuple(
      Enum.reverse(
        window_dims(Tuple.to_list(shape), Tuple.to_list(window), Tuple.to_list(strides), [])
      )
    )
  end

  defp window_dims([], [], [], acc), do: acc

  defp window_dims([dim | shape], [w | window], [s | strides], acc),
    do: window_dims(shape, window, strides, [max(div(dim - w, s) + 1, 1) | acc])

  # Ensures the window is valid given the shape.
  # A window is valid as long as it's rank matches
  # the rank of the given shape.
  defp validate_window!(shape, window)

  defp validate_window!(shape, window) when tuple_size(shape) != tuple_size(window),
    do:
      raise(
        ArgumentError,
        "invalid window dimensions, rank of shape (#{tuple_size(shape)})" <>
          " does not match rank of window (#{tuple_size(window)})"
      )

  defp validate_window!(_, _), do: :ok

  # Ensures the strides are valid given the shape.
  # A stride is valid as long as it's rank matches
  # the rank of the given shape.
  defp validate_strides!(shape, strides)

  defp validate_strides!(shape, strides) when tuple_size(strides) != tuple_size(shape),
    do:
      raise(
        ArgumentError,
        "invalid stride dimensions, rank of shape (#{tuple_size(shape)})" <>
          " does not match rank of stride (#{tuple_size(strides)})"
      )

  defp validate_strides!(_, _), do: :ok

  @doc """
  Output shape after a squeeze operation.

  ## Examples

      iex> Nx.Shape.squeeze({2, 1, 1}, [1, 2])
      {2}

      iex> Nx.Shape.squeeze({1, 2}, [0])
      {2}

  ### Error cases

      iex> Nx.Shape.squeeze({2, 2, 1}, [1])
      ** (ArgumentError) cannot squeeze dimensions whose sizes are not 1, got 2 for dimension 1
  """
  def squeeze(shape, axes) do
    List.to_tuple(Enum.reverse(squeeze_dims(Enum.with_index(Tuple.to_list(shape)), axes, [])))
  end

  defp squeeze_dims([], _, acc), do: acc

  defp squeeze_dims([{s, i} | shape], axes, acc) do
    if i in axes do
      if s == 1 do
        squeeze_dims(shape, axes, acc)
      else
        raise ArgumentError,
              "cannot squeeze dimensions whose sizes are not 1, got #{s} for dimension #{i}"
      end
    else
      squeeze_dims(shape, axes, [s | acc])
    end
  end

  @doc """
  Output shape after a padding operation.

  ## Examples

      iex> Nx.Shape.pad({3, 2, 4}, [{0, 1}, {1, 2}, {1, 1}])
      {4, 5, 6}

      iex> Nx.Shape.pad({}, [])
      {}

      iex> Nx.Shape.pad({2, 2}, [{1, 1}, {0, 0}])
      {4, 2}

  ### Error cases

      iex> Nx.Shape.pad({2, 2, 3}, [{0, 1}, {1, 2}])
      ** (ArgumentError) invalid padding configuration, rank of padding configuration and shape must match
  """
  def pad(shape, padding_config) do
    shape
    |> Tuple.to_list()
    |> padded_dims(padding_config, [])
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp padded_dims([], [], acc), do: acc

  defp padded_dims([_ | _], [], _acc),
    do:
      raise(
        ArgumentError,
        "invalid padding configuration, rank of padding configuration" <>
          " and shape must match"
      )

  defp padded_dims([], [_ | _], _acc),
    do:
      raise(
        ArgumentError,
        "invalid padding configuration, rank of padding configuration" <>
          " and shape must match"
      )

  defp padded_dims([s | shape], [{edge_low, edge_high} | config], acc),
    do: padded_dims(shape, config, [s + edge_low + edge_high | acc])

  ## Axes helpers

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
          "given axis (#{inspect(axis)}) invalid for shape with rank #{tuple_size(shape)}"
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
    rank = tuple_size(shape)
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
    min_size = tuple_size(shape)
    max_size = tuple_size(new_shape)
    count_up(min_size, max_size - min_size)
  end

  @doc """
  Returns the axes for squeezing.

  ## Examples

      iex> Nx.Shape.squeeze_axes({2, 1, 1})
      [1, 2]

      iex> Nx.Shape.squeeze_axes({1, 2, 1, 3, 2, 1})
      [0, 2, 5]
  """
  def squeeze_axes(shape) do
    for {1, i} <- Enum.with_index(Tuple.to_list(shape)), do: i
  end

  ## Helpers

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  defp count_down(0, _n), do: []
  defp count_down(i, n), do: [n | count_down(i - 1, n - 1)]
end
