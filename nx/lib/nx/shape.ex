defmodule Nx.Shape do
  # Conveniences for manipulating shapes internal to Nx.
  @moduledoc false

  @doc """
  Validates a given shape with `kind`.

  ## Examples

      iex> Nx.Shape.validate!({1, 2, 3}, :window_dimensions)
      {1, 2, 3}

      iex> Nx.Shape.validate!({0, 2, 3}, :window_dimensions)
      ** (ArgumentError) invalid dimension in axis 0 in window_dimensions. Each dimension must be a positive integer, got 0 in shape {0, 2, 3}

  """
  def validate!(shape, kind) when is_tuple(shape) do
    validate!(shape, tuple_size(shape), kind)
  end

  def validate!(other, kind) do
    raise ArgumentError,
          "invalid #{kind}. #{kind} is a n-element tuple with the size of each dimension. " <>
            "Got: #{inspect(other)}"
  end

  defp validate!(shape, 0, _kind), do: shape

  defp validate!(shape, pos, kind) do
    dim = :erlang.element(pos, shape)

    if is_integer(dim) and dim > 0 do
      validate!(shape, pos - 1, kind)
    else
      raise ArgumentError,
            "invalid dimension in axis #{pos - 1} in #{kind}. Each dimension must be a positive integer, " <>
              "got #{inspect(dim)} in shape #{inspect(shape)}"
    end
  end

  @doc """
  Converts a shape to an algebra document for inspection.
  """
  def to_algebra(shape, names, open, close) do
    shape
    |> Tuple.to_list()
    |> Enum.zip_with(names, fn
      number, nil ->
        Inspect.Algebra.concat([open, Integer.to_string(number), close])

      number, name ->
        Inspect.Algebra.concat([
          open,
          Atom.to_string(name),
          ": ",
          Integer.to_string(number),
          close
        ])
    end)
    |> Inspect.Algebra.concat()
  end

  @doc """
  Converts shape and name to a string.

  ## Examples

      iex> Nx.Shape.to_string({1, 2, 3}, [:foo, nil, :bat])
      "[foo: 1][2][bat: 3]"

  """
  def to_string(shape, names) do
    to_algebra(shape, names, "[", "]")
    |> Inspect.Algebra.format(:infinity)
    |> IO.iodata_to_binary()
  end

  @doc """
  Validates the names of axes.
  """
  def named_axes!(names, shape) do
    n_dims = tuple_size(shape)

    if names do
      n_names = length(names)

      if n_names != n_dims do
        raise ArgumentError,
              "invalid names for tensor of rank #{n_dims}," <>
                " when specifying names every dimension must" <>
                " have a name or be nil"
      else
        names
      end
    else
      List.duplicate(nil, n_dims)
    end
  end

  @doc """
  Finds the axis for the given name.
  """
  def find_name!(names, name) do
    Enum.find_index(names, &(&1 == name)) ||
      raise(
        ArgumentError,
        "tensor does not have name #{inspect(name)}. The tensor names are: #{inspect(names)}"
      )
  end

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

  ## Examples

  ### Scalar Shapes

      iex> Nx.Shape.binary_broadcast({}, [], {}, [])
      {{}, []}
      iex> Nx.Shape.binary_broadcast({}, [], {4, 2, 1, 5}, [:batch, nil, :data, nil])
      {{4, 2, 1, 5}, [:batch, nil, :data, nil]}

  ### n-D Shapes

      iex> Nx.Shape.binary_broadcast({8, 1, 6, 1}, [:batch, nil, :data, nil], {7, 1, 5}, [:time, :data, nil])
      {{8, 7, 6, 5}, [:batch, :time, :data, nil]}
      iex> Nx.Shape.binary_broadcast({7, 1, 5}, [:time, :data, nil], {8, 1, 6, 1},  [:batch, nil, :data, nil])
      {{8, 7, 6, 5}, [:batch, :time, :data, nil]}
      iex> Nx.Shape.binary_broadcast({5, 4}, [nil, nil], {1}, [:data])
      {{5, 4}, [nil, :data]}
      iex> Nx.Shape.binary_broadcast({3, 1}, [:x, :y], {15, 3, 5}, [:batch, :x, nil])
      {{15, 3, 5}, [:batch, :x, :y]}

  ### Error cases

      iex> Nx.Shape.binary_broadcast({4, 2, 5}, [nil, nil, nil], {3, 2, 5}, [:batch, :x, :y])
      ** (ArgumentError) cannot broadcast tensor of dimensions {4, 2, 5} to {3, 2, 5}

      iex> Nx.Shape.binary_broadcast({1, 2, 5}, [:batch, :x, :y], {3, 2, 5}, [:time, :x, :y])
      ** (ArgumentError) cannot merge names :batch, :time
  """
  def binary_broadcast(left_shape, left_names, right_shape, right_names)

  def binary_broadcast(shape, names, shape, names), do: {shape, names}

  def binary_broadcast(left_shape, left_names, right_shape, right_names)
      when is_tuple(left_shape) and is_tuple(right_shape) do
    left_rank = tuple_size(left_shape)
    right_rank = tuple_size(right_shape)
    rank = max(left_rank, right_rank)

    left_lower_and_names =
      shape_and_names_to_lower_ranked_list(
        left_shape,
        Enum.reverse(left_names),
        left_rank,
        rank
      )

    right_lower_and_names =
      shape_and_names_to_lower_ranked_list(
        right_shape,
        Enum.reverse(right_names),
        right_rank,
        rank
      )

    {left_lower, left_names} = Enum.unzip(left_lower_and_names)
    {right_lower, right_names} = Enum.unzip(right_lower_and_names)

    case binary_broadcast(left_lower, left_names, right_lower, right_names, [], []) do
      {:ok, new_shape, new_names} ->
        {new_shape, new_names}

      :error ->
        raise ArgumentError,
              "cannot broadcast tensor of dimensions #{inspect(left_shape)} " <>
                "to #{inspect(right_shape)}"
    end
  end

  defp binary_broadcast(
         [ldim | ldims],
         [lname | lnames],
         [rdim | rdims],
         [rname | rnames],
         shape_acc,
         names_acc
       )
       when rdim == 1 or ldim == 1 or rdim == ldim do
    names_acc = [merge_names!(lname, rname) | names_acc]
    binary_broadcast(ldims, lnames, rdims, rnames, [max(rdim, ldim) | shape_acc], names_acc)
  end

  defp binary_broadcast([], [], [], [], shape_acc, names_acc),
    do: {:ok, List.to_tuple(shape_acc), names_acc}

  defp binary_broadcast(_, _, _, _, _, _),
    do: :error

  defp shape_and_names_to_lower_ranked_list(_tuple, _names, 0, 0),
    do: []

  defp shape_and_names_to_lower_ranked_list(tuple, [], 0, rank),
    do: [{1, nil} | shape_and_names_to_lower_ranked_list(tuple, [], 0, rank - 1)]

  defp shape_and_names_to_lower_ranked_list(tuple, [n | names], size, rank),
    do: [
      {:erlang.element(size, tuple), n}
      | shape_and_names_to_lower_ranked_list(tuple, names, size - 1, rank - 1)
    ]

  @doc """
  Contracts a shape along the given axes.

  It expects the axes to have been normalized.

  ## Examples

      iex> Nx.Shape.contract({4, 1, 2}, [1], [:batch, :x, :y], false)
      {{4, 2}, [:batch, :y]}

      iex> Nx.Shape.contract({2, 4, 6, 5}, [1, 3], [:batch, :x, :y, :z], false)
      {{2, 6}, [:batch, :y]}

      iex> Nx.Shape.contract({1, 2, 3}, [], [:batch, :x, :y], false)
      {{1, 2, 3}, [:batch, :x, :y]}

      iex> Nx.Shape.contract({4, 2, 8}, [2], [:x, :y, :z], false)
      {{4, 2}, [:x, :y]}

      iex> Nx.Shape.contract({4, 2, 8}, [2], [:x, :y, :z], true)
      {{4, 2, 1}, [:x, :y, :z]}

  """
  def contract(shape, axes, names, keep_axes) do
    {new_shape, new_names} =
      Enum.unzip(contract(shape, axes, names, 0, tuple_size(shape), keep_axes))

    {List.to_tuple(new_shape), new_names}
  end

  defp contract(_shape, _axes, _names, n, n, _keep_axes) do
    []
  end

  defp contract(shape, axes, [name | names], i, n, keep_axes) do
    cond do
      i not in axes ->
        [{elem(shape, i), name} | contract(shape, axes, names, i + 1, n, keep_axes)]

      keep_axes ->
        [{1, name} | contract(shape, axes, names, i + 1, n, keep_axes)]

      true ->
        contract(shape, axes, names, i + 1, n, keep_axes)
    end
  end

  @doc """
  Transposes a shape according to the given permutation.

  ## Examples

    iex> Nx.Shape.transpose({4, 8, 2, 1}, [1, 0, 3, 2], [:batch, :channels, :height, :width])
    {{8, 4, 1, 2}, [:channels, :batch, :width, :height]}

  ### Error cases

    iex> Nx.Shape.transpose({4, 8, 2, 1}, [0, 1, 2], [:batch, nil, nil, nil])
    ** (ArgumentError) expected length of permutation (3) to match rank of shape (4)

  """
  def transpose(shape, permutation, names)

  def transpose(shape, permutation, names) when tuple_size(shape) == length(permutation) do
    {new_shape, new_names} =
      Enum.unzip(Enum.map(permutation, &{elem(shape, &1), Enum.at(names, &1)}))

    {List.to_tuple(new_shape), new_names}
  end

  def transpose(shape, permutation, _names) do
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

      iex> Nx.Shape.zip_reduce({1, 2, 3}, [0, 1], [:batch, :x, :y], {3, 1, 2}, [1, 2], [:batch, :x, :y])
      {{3, 3}, [:y, :batch]}

      iex> Nx.Shape.zip_reduce({1, 2, 3}, [0, 1], [nil, nil, nil], {1, 2, 3}, [1, 2], [nil, nil, nil])
      ** (ArgumentError) dot/zip expects shapes to be compatible, dimension 0 of left-side (1) does not equal dimension 1 of right-side (2)

      iex> Nx.Shape.zip_reduce({2, 2}, [1], [:x, :y], {2, 2}, [0], [:y, :x])
      ** (ArgumentError) operation would result in duplicate names [:x, :x], please rename your tensors to avoid duplicates
  """
  def zip_reduce(s1, axes1, names1, s2, axes2, names2) do
    validate_zip_reduce_axes!(s1, axes1, s2, axes2)
    {l1, n1} = Enum.unzip(contract(s1, axes1, names1, 0, tuple_size(s1), false))
    {l2, n2} = Enum.unzip(contract(s2, axes2, names2, 0, tuple_size(s2), false))
    new_names = n1 ++ n2

    non_nil_names = Enum.filter(new_names, &(&1 != nil))

    if length(non_nil_names) != length(Enum.uniq(non_nil_names)),
      do:
        raise(
          ArgumentError,
          "operation would result in duplicate names #{inspect(new_names)}," <>
            " please rename your tensors to avoid duplicates"
        )

    {List.to_tuple(l1 ++ l2), n1 ++ n2}
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
  Returns a padding configuration based on the given pad mode
  for the given input shape, kernel size and stride.

  By default, interior padding is not considered in the padding
  configuration.

  ## Examples

      iex> Nx.Shape.to_padding_config({2, 3, 2}, {2, 3, 2}, [1, 1, 1], :valid)
      [{0, 0}, {0, 0}, {0, 0}]

      iex> Nx.Shape.to_padding_config({2, 3, 2}, {2, 3, 2}, [1, 1, 1], :valid, true)
      [{0, 0, 0}, {0, 0, 0}, {0, 0, 0}]

      iex> Nx.Shape.to_padding_config({12, 12}, {2, 2}, [1, 1], :same)
      [{0, 1}, {0, 1}]

  ### Error cases

      iex> Nx.Shape.to_padding_config({2, 3, 2}, {2, 3, 2}, [1, 1, 2], :foo)
      ** (ArgumentError) invalid padding mode specified, padding must be one of :valid, :same, or a padding configuration, got: :foo

  """
  def to_padding_config(shape, kernel_size, strides, mode, interior \\ false) do
    case mode do
      :valid ->
        pad_valid(shape, kernel_size, strides, interior)

      :same ->
        pad_same(shape, kernel_size, strides, interior)

      config when is_list(config) ->
        config

      mode ->
        raise ArgumentError,
              "invalid padding mode specified, padding must be one" <>
                " of :valid, :same, or a padding configuration, got:" <>
                " #{inspect(mode)}"
    end
  end

  defp pad_valid(shape, _, _, interior) do
    if interior,
      do: List.duplicate({0, 0, 0}, Nx.rank(shape)),
      else: List.duplicate({0, 0}, Nx.rank(shape))
  end

  defp pad_same(shape, kernel_size, strides, interior) do
    Enum.zip_with([Tuple.to_list(shape), Tuple.to_list(kernel_size), strides], fn [dim, k, s] ->
      padding_size = max((dim - 1) * s + k - dim, 0)
      lo = floor(padding_size / 2)
      hi = ceil(padding_size / 2)
      if interior, do: {lo, hi, 0}, else: {lo, hi}
    end)
  end

  @doc """
  Dilates the given input shape according to dilation.

  ## Examples

      iex> Nx.Shape.dilate({3, 3, 3}, [1, 2, 1])
      {3, 5, 3}

      iex> Nx.Shape.dilate({2, 4, 2}, [3, 1, 3])
      {4, 4, 4}
  """
  def dilate(shape, dilation) when is_tuple(shape) and is_list(dilation) do
    unless Enum.all?(dilation, &(&1 >= 1)) do
      raise ArgumentError,
            "dilation rates must be greater than or equal to 1" <>
              " got #{inspect(dilation)}"
    end

    dilated_padding_config = Enum.map(dilation, fn x -> {0, 0, x - 1} end)
    pad(shape, dilated_padding_config)
  end

  @doc """
  Output shape after a convolution.
  """
  def conv(
        input_shape,
        input_names,
        kernel_shape,
        kernel_names,
        strides,
        padding,
        feature_group_count,
        batch_group_count,
        input_dilation,
        kernel_dilation,
        input_permutation,
        kernel_permutation,
        output_permutation
      ) do
    validate_conv_ranks!(input_shape, kernel_shape)
    validate_conv_strides!(input_shape, strides)
    validate_conv_dilations!(input_shape, kernel_shape, input_dilation, kernel_dilation)

    {input_shape, permuted_input_names} = transpose(input_shape, input_permutation, input_names)
    input_shape = dilate(input_shape, [1, 1 | input_dilation])

    {kernel_shape, _} = transpose(kernel_shape, kernel_permutation, kernel_names)
    kernel_shape = dilate(kernel_shape, [1, 1 | kernel_dilation])

    validate_conv_groups!(input_shape, kernel_shape, feature_group_count, batch_group_count)

    num_filters = elem(kernel_shape, 0)
    batch_size = elem(input_shape, 0)

    filter_shape =
      kernel_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    spatial_dims =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    padding_config =
      to_padding_config(
        spatial_dims,
        filter_shape,
        List.duplicate(1, Nx.rank(spatial_dims)),
        padding
      )

    old_spatial_dims =
      spatial_dims
      |> pad(Enum.map(padding_config, fn {x, y} -> {x, y, 0} end))
      |> Tuple.to_list()

    spatial_dims = do_conv_spatial_dims(old_spatial_dims, Tuple.to_list(filter_shape), strides)
    shape = List.to_tuple([div(batch_size, batch_group_count), num_filters | spatial_dims])

    inv_output_permutation =
      output_permutation
      |> Enum.with_index()
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))

    {shape, names} = transpose(shape, inv_output_permutation, permuted_input_names)

    {shape, names, padding_config}
  end

  defp validate_conv_ranks!(input_shape, kernel_shape) do
    cond do
      Nx.rank(input_shape) < 3 ->
        raise ArgumentError,
              "input shape in conv requires at least rank 3," <>
                " shape #{inspect(input_shape)} has rank #{Nx.rank(input_shape)}"

      Nx.rank(kernel_shape) < 3 ->
        raise ArgumentError,
              "kernel shape in conv requires at least rank 3," <>
                " shape #{inspect(kernel_shape)} has rank #{Nx.rank(kernel_shape)}"

      true ->
        :ok
    end
  end

  defp validate_conv_strides!(input_shape, strides) do
    if length(strides) != Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "rank of strides much match rank of spatial dimensions" <>
              " got strides #{inspect(strides)} with rank #{length(strides)}" <>
              " and got input shape #{inspect(input_shape)} of rank" <>
              " #{Nx.rank(input_shape) - 2}"
    end
  end

  # Validates the input and kernel dilations given to Nx.conv
  defp validate_conv_dilations!(input_shape, kernel_shape, input_dilation, kernel_dilation) do
    cond do
      is_list(input_dilation) and length(input_dilation) != Nx.rank(input_shape) - 2 ->
        raise ArgumentError,
              "must specify dilation for each spatial dimension of the input" <>
                " or specify an integer dilation factor"

      is_list(input_dilation) and Enum.any?(input_dilation, &(&1 < 1 || !is_integer(&1))) ->
        raise ArgumentError,
              "input dilation of each dimension must be a positive integer, got " <>
                inspect(input_dilation)

      is_list(kernel_dilation) and length(kernel_dilation) != Nx.rank(kernel_shape) - 2 ->
        raise ArgumentError,
              "must specify dilation for each spatial dimension of the kernel" <>
                " or specify an integer dilation factor"

      is_list(kernel_dilation) and Enum.any?(kernel_dilation, &(&1 < 1 || !is_integer(&1))) ->
        raise ArgumentError,
              "kernel dilation of each dimension must be a positive integer, got " <>
                inspect(kernel_dilation)

      true ->
        :ok
    end
  end

  defp validate_conv_groups!(input_shape, kernel_shape, feature_groups, batch_groups) do
    tensor_input_batch_size = elem(input_shape, 0)
    tensor_input_channels = elem(input_shape, 1)
    kernel_input_channels = elem(kernel_shape, 1)
    kernel_output_channels = elem(kernel_shape, 0)

    cond do
      batch_groups != 1 and feature_groups != 1 ->
        raise ArgumentError,
              "either batch groups or feature groups must be 1," <>
                " got batch_groups = #{batch_groups} and feature_groups = #{feature_groups}"

      rem(tensor_input_batch_size, batch_groups) != 0 ->
        raise ArgumentError,
              "batch groups must evenly divide input batch size" <>
                " got rem(#{batch_groups}, #{tensor_input_batch_size}) != 0"

      rem(kernel_output_channels, feature_groups) != 0 ->
        raise ArgumentError,
              "size of kernel output channels must be evenly divisible by feature groups" <>
                " got rem(#{kernel_output_channels}, #{feature_groups}) != 0 for kernel" <>
                " with shape #{inspect(kernel_shape)}"

      rem(kernel_output_channels, batch_groups) != 0 ->
        raise ArgumentError,
              "size of kernel output channels must be evenly divisible by batch groups" <>
                " got rem(#{kernel_output_channels}, #{batch_groups}) != 0 for kernel" <>
                " with shape #{inspect(kernel_shape)}"

      tensor_input_channels != kernel_input_channels * feature_groups ->
        raise ArgumentError,
              "size of input channels divided by feature groups must match size of kernel channels," <>
                " got #{tensor_input_channels} / #{feature_groups} != #{kernel_input_channels}" <>
                " for shapes #{inspect(input_shape)} and #{inspect(kernel_shape)}"

      true ->
        :ok
    end
  end

  defp do_conv_spatial_dims([], [], []), do: []

  defp do_conv_spatial_dims([cur | spatial], [f | filters], [s | strides]),
    do: [floor((cur - f) / s) + 1 | do_conv_spatial_dims(spatial, filters, strides)]

  @doc """
  Output shape after a pooling or reduce window operation.

  ## Examples

    iex> Nx.Shape.pool({3, 3}, {1, 2}, [1, 1], :valid, [1, 1])
    {{3, 2}, [{0, 0}, {0, 0}]}

    iex> Nx.Shape.pool({3, 2, 3}, {2, 1, 1}, [1, 2, 1], :same, [1, 1, 1])
    {{3, 1, 3}, [{0, 1}, {0, 0}, {0, 0}]}

  ### Error cases

    iex> Nx.Shape.pool({1, 2, 3}, {2, 1, 1}, [1, 1, 1], :valid, [1, 1, 1])
    ** (ArgumentError) window dimensions would result in empty tensor which is not currently supported in Nx, please open an issue if you'd like this behavior to change

    iex> Nx.Shape.pool({1, 2, 3}, {2, 1}, [1, 1, 1], :valid, [1, 1, 1])
    ** (ArgumentError) invalid window dimensions, rank of shape (3) does not match rank of window (2)

    iex> Nx.Shape.pool({1, 2, 3}, {2, 1, 1}, [1, 1], :valid, [1, 1, 1])
    ** (ArgumentError) invalid stride dimensions, rank of shape (3) does not match rank of stride (2)
  """
  def pool(shape, kernel_size, strides, padding, kernel_dilation) do
    validate_window!(shape, kernel_size)
    validate_strides!(shape, strides)

    kernel_size = dilate(kernel_size, kernel_dilation)

    padding_config =
      to_padding_config(shape, kernel_size, List.duplicate(1, Nx.rank(shape)), padding)

    shape = pad(shape, Enum.map(padding_config, fn {x, y} -> {x, y, 0} end))

    {List.to_tuple(do_pool(strides, shape, kernel_size, 0)), padding_config}
  end

  defp do_pool([], _shape, _window, _pos), do: []

  defp do_pool([s | strides], shape, window, pos) do
    dim = elem(shape, pos)
    w = elem(window, pos)
    new_dim = div(dim - w, s) + 1

    if new_dim <= 0 do
      raise ArgumentError,
            "window dimensions would result in empty tensor" <>
              " which is not currently supported in Nx, please" <>
              " open an issue if you'd like this behavior to change"
    end

    [new_dim | do_pool(strides, shape, window, pos + 1)]
  end

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

  defp validate_strides!(shape, strides) when tuple_size(shape) != length(strides),
    do:
      raise(
        ArgumentError,
        "invalid stride dimensions, rank of shape (#{tuple_size(shape)})" <>
          " does not match rank of stride (#{length(strides)})"
      )

  defp validate_strides!(_, _), do: :ok

  @doc """
  Output shape after a squeeze operation.

  ## Examples

      iex> Nx.Shape.squeeze({2, 1, 1}, [1, 2], [:batch, :x, :y])
      {{2}, [:batch]}

      iex> Nx.Shape.squeeze({1, 2}, [0], [:batch, :x])
      {{2}, [:x]}

  ### Error cases

      iex> Nx.Shape.squeeze({2, 2, 1}, [1], [:batch, :x, :y])
      ** (ArgumentError) cannot squeeze dimensions whose sizes are not 1, got 2 for dimension 1

  """
  def squeeze(shape, axes, names) do
    squeeze(Enum.with_index(Tuple.to_list(shape)), axes, names, [], [])
  end

  defp squeeze([], _, _, sacc, nacc) do
    {List.to_tuple(Enum.reverse(sacc)), Enum.reverse(nacc)}
  end

  defp squeeze([{s, i} | shape], axes, [n | names], sacc, nacc) do
    if i in axes do
      if s == 1 do
        squeeze(shape, axes, names, sacc, nacc)
      else
        raise ArgumentError,
              "cannot squeeze dimensions whose sizes are not 1, got #{s} for dimension #{i}"
      end
    else
      squeeze(shape, axes, names, [s | sacc], [n | nacc])
    end
  end

  @doc """
  Output shape after a padding operation.

  ## Examples

      iex> Nx.Shape.pad({3, 2, 4}, [{0, 1, 0}, {1, 2, 0}, {1, 1, 0}])
      {4, 5, 6}

      iex> Nx.Shape.pad({}, [])
      {}

      iex> Nx.Shape.pad({2, 2}, [{1, 1, 0}, {0, 0, 0}])
      {4, 2}

      iex> Nx.Shape.pad({2, 3}, [{0, 0, 1}, {0, 0, 1}])
      {3, 5}

  ### Error cases

      iex> Nx.Shape.pad({2, 2, 3}, [{0, 1, 0}, {1, 2, 0}])
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

  defp padded_dims([s | shape], [{edge_low, edge_high, interior} | config], acc) do
    interior_padding_factor = (s - 1) * interior
    padded_dims(shape, config, [s + interior_padding_factor + edge_low + edge_high | acc])
  end

  ## Axes helpers

  @doc """
  Normalize the axis to the given shape.

  ## Examples

      iex> Nx.Shape.normalize_axis({4, 2, 3}, -1, [:batch, :x, :y])
      2

      iex> Nx.Shape.normalize_axis({4, 2, 1, 4}, -2, [:batch, :x, :y, :z])
      2

      iex> Nx.Shape.normalize_axis({4, 2, 1, 4}, 1, [:batch, :x, :y, :z])
      1

      iex> Nx.Shape.normalize_axis({4, 2, 1, 4}, :z, [:batch, :x, :y, :z])
      3

  ### Error cases

      iex> Nx.Shape.normalize_axis({4, 2, 5}, -4, [:batch, :x, :y])
      ** (ArgumentError) given axis (-4) invalid for shape with rank 3

      iex> Nx.Shape.normalize_axis({4, 2, 5}, 3, [:batch, :x, :y])
      ** (ArgumentError) given axis (3) invalid for shape with rank 3

      iex> Nx.Shape.normalize_axis({4, 2, 5}, :z, [:batch, :x, :y])
      ** (ArgumentError) key :z not found in tensor with names [:batch, :x, :y]

      iex> Nx.Shape.normalize_axis({4, 2, 5}, nil, [:batch, nil, nil])
      ** (ArgumentError) axis name cannot be nil

  """
  def normalize_axis(shape, axis, names)

  def normalize_axis(shape, axis, _names) when axis < 0 and abs(axis) <= tuple_size(shape),
    do: tuple_size(shape) + axis

  def normalize_axis(shape, axis, _names) when axis >= 0 and axis < tuple_size(shape),
    do: axis

  def normalize_axis(_shape, nil, _names),
    do: raise(ArgumentError, "axis name cannot be nil")

  def normalize_axis(_shape, axis, names) when is_atom(axis) do
    if axis in names do
      Enum.with_index(names)[axis]
    else
      raise ArgumentError, "key #{inspect(axis)} not found in tensor with names #{inspect(names)}"
    end
  end

  def normalize_axis(shape, axis, _names) do
    raise ArgumentError,
          "given axis (#{inspect(axis)}) invalid for shape with rank #{tuple_size(shape)}"
  end

  @doc """
  Normalize a list of unique axis.

  See `normalize_axis/1`.

  ## Examples

      iex> Nx.Shape.normalize_axes({2, 3, 4}, [-1, 0], [:batch, nil])
      [2, 0]

      iex> Nx.Shape.normalize_axes({2, 3, 4}, [:batch, 1], [:batch, :x])
      [0, 1]

  ### Error Cases

      iex> Nx.Shape.normalize_axes({2, 3, 4}, [1, 1], [nil, nil, nil])
      ** (ArgumentError) axes [1, 1] must be unique integers between 0 and 2
  """
  def normalize_axes(shape, axes, names) when is_list(axes) do
    normalized = Enum.map(axes, &normalize_axis(shape, &1, names))

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

  @doc """
  Returns the shape after a slice.

  ## Examples

      iex> Nx.Shape.slice({2, 15, 30}, [1, 4, 10], [1, 1, 10], [1, 1, 3])
      {1, 1, 4}

  ### Error cases

      iex> Nx.Shape.slice({2, 15, 30}, [1, 4, 10], [3, 1, 1], [1, 1, 1])
      ** (ArgumentError) length at axis 0 must be less than axis size of 2, got: 3

  """
  def slice(shape, start_indices, lengths, strides) do
    rank = Nx.rank(shape)

    if length(strides) != rank do
      raise ArgumentError, "invalid strides rank for shape of rank #{rank}"
    end

    if length(start_indices) != rank do
      raise ArgumentError, "invalid start indices rank for shape of rank #{rank}"
    end

    if length(lengths) != rank do
      raise ArgumentError, "invalid limit indices rank for shape of rank #{rank}"
    end

    shape
    |> do_slice(0, lengths, strides)
    |> List.to_tuple()
  end

  defp do_slice(shape, pos, [len | lengths], [s | strides]) do
    dim = elem(shape, pos)

    if not is_integer(len) or len < 1 do
      raise ArgumentError,
            "length at axis #{pos} must be greater than or equal to 1, got: #{inspect(len)}"
    end

    if not is_integer(s) or s < 1 do
      raise ArgumentError,
            "stride at axis #{pos} must be greater than or equal to 1, got: #{inspect(s)}"
    end

    if len > dim do
      raise ArgumentError,
            "length at axis #{pos} must be less than axis size of #{dim}, got: #{len}"
    end

    [Kernel.ceil(len / s) | do_slice(shape, pos + 1, lengths, strides)]
  end

  defp do_slice(_shape, _pos, [], []), do: []

  @doc """
  Returns the shape and names after a put_slice.

  ## Examples

      iex> Nx.Shape.put_slice({2, 3}, [nil, :data], {1, 2}, [:batch, nil], [1, 1])
      {{2, 3}, [:batch, :data]}

      iex> Nx.Shape.put_slice({2, 3}, [nil, nil], {2, 3}, [nil, nil], [0, 1])
      {{2, 3}, [nil, nil]}

  """
  def put_slice(shape, names, slice_shape, slice_names, start_indices) do
    rank = Nx.rank(shape)

    if length(start_indices) != rank do
      raise ArgumentError, "invalid start indices rank for shape of rank #{rank}"
    end

    if Nx.rank(slice_shape) != rank do
      raise ArgumentError,
            "invalid slice for put_slice, rank of slice must match #{rank}, " <>
              "got: #{Nx.rank(slice_shape)}"
    end

    shape
    |> Tuple.to_list()
    |> do_put_slice(names, Tuple.to_list(slice_shape), slice_names, [])
    |> case do
      :error ->
        raise ArgumentError,
              "slice shape #{inspect(slice_shape)} must be less than or equal to " <>
                "tensor shape #{inspect(shape)}"

      names ->
        {shape, names}
    end
  end

  defp do_put_slice([s | _], _, [slice | _], _, _) when slice > s do
    :error
  end

  defp do_put_slice([_ | shape], [n | names], [_ | s_shape], [s_name | s_names], acc) do
    do_put_slice(shape, names, s_shape, s_names, [merge_names!(n, s_name) | acc])
  end

  defp do_put_slice([], [], [], [], acc), do: Enum.reverse(acc)

  @doc """
  Returns the shape and names after a concat.

  ## Examples

      iex> Nx.Shape.concatenate([{2, 3, 2}, {1, 3, 2}, {4, 3, 2}], [[:x, :y, :z], [:x, :y, :z], [:x, :y, :z]], 0)
      {{7, 3, 2}, [:x, :y, :z]}
  """
  def concatenate(shapes, names, axis) do
    names = validate_concat_names!(names)
    {concat_dims(shapes, axis), names}
  end

  defp concat_dims([s1 | shapes], axis) do
    s1 = Tuple.to_list(s1)

    shapes
    |> Enum.reduce(s1, &concat_shapes(Tuple.to_list(&1), &2, axis))
    |> List.to_tuple()
  end

  defp concat_shapes(shape1, shape2, axis) do
    shape1
    |> Enum.zip(shape2)
    |> Enum.with_index(fn {s1, s2}, i ->
      cond do
        i == axis ->
          s1 + s2

        s1 == s2 ->
          s1

        true ->
          raise ArgumentError,
                "non-concat dims must be equal got" <>
                  " #{inspect(s1)} and #{inspect(s2)}" <>
                  " while concatenating on axis #{axis}"
      end
    end)
  end

  @doc """
  Calculates the intermediate and final shapes used by the
  `Nx.tile` function.
  """
  def tile(%Nx.Tensor{shape: old_shape}, repetitions) do
    num_dims = tuple_size(old_shape)
    length_reps = length(repetitions)

    # grow the dimensionality of the tensor by new_dims_count
    shape_grow_count = Kernel.max(length_reps - num_dims, 0)
    resized_shape_list = List.duplicate(1, shape_grow_count) ++ Tuple.to_list(old_shape)

    repetitions_grow_count = Kernel.max(num_dims - length_reps, 0)
    resized_repetitions = List.duplicate(1, repetitions_grow_count) ++ repetitions

    broadcast_shape = resized_repetitions |> alternate(resized_shape_list) |> List.to_tuple()

    tensor_reshape =
      [1 | Enum.intersperse(resized_shape_list, 1)]
      |> List.to_tuple()

    result_shape =
      resized_repetitions
      |> Enum.zip(resized_shape_list)
      |> Enum.map(fn {x, y} -> x * y end)
      |> List.to_tuple()

    {tensor_reshape, broadcast_shape, result_shape}
  end

  defp alternate([], []), do: []
  defp alternate([h1 | tl1], [h2 | tl2]), do: [h1, h2 | alternate(tl1, tl2)]

  @doc """
  Calculates the output shape of a dot product.
  """
  def dot(s1, c1, names1, b1, s2, c2, names2, b2) do
    validate_dot_axes!(s1, c1, b1, s2, c2, b2)

    {batch_dims, batch_names, s1, c1, names1, s2, c2, names2} =
      prep_dot_batch_output(s1, c1, names1, b1, s2, c2, names2, b2)

    # zip reduce without the batched dimensions
    {output_shape, output_names} = zip_reduce(s1, c1, names1, s2, c2, names2)

    # re-add the batched dimensions.
    if is_nil(batch_dims) do
      {output_shape, output_names}
    else
      output_shape =
        Enum.reduce(Enum.reverse(batch_dims), output_shape, fn x, acc ->
          Tuple.insert_at(acc, 0, x)
        end)

      output_names = batch_names ++ output_names
      {output_shape, output_names}
    end
  end

  defp prep_dot_batch_output(s1, c1, names1, b1, s2, c2, names2, b2) do
    case {b1, b2} do
      {[], []} ->
        {nil, nil, s1, c1, names1, s2, c2, names2}

      {b1, b2} ->
        batch_dims = Enum.map(b1, &elem(s1, &1))
        batch_names = Enum.map(b1, &Enum.at(names1, &1))
        {s1, c1, names1} = shift_left_for_batch(s1, c1, b1, names1)
        {s2, c2, names2} = shift_left_for_batch(s2, c2, b2, names2)
        {batch_dims, batch_names, s1, c1, names1, s2, c2, names2}
    end
  end

  defp shift_left_for_batch(shape, contract_axes, batch_axes, names) do
    non_batch_shapes =
      batch_axes
      |> Enum.reduce(shape, fn _, acc -> Tuple.delete_at(acc, 0) end)

    contract_axes = shift_left_axes(contract_axes, length(batch_axes))

    names =
      batch_axes
      |> Enum.reduce(names, fn _, [_ | tail] -> tail end)

    {non_batch_shapes, contract_axes, names}
  end

  defp shift_left_axes(axes, num_batch_dims) do
    Enum.map(axes, fn a -> a - num_batch_dims end)
  end

  defp validate_dot_axes!(s1, c1, b1, s2, c2, b2) do
    left_batched? = b1 != []
    right_batched? = b2 != []

    if not left_batched? and right_batched? do
      raise ArgumentError, "left tensor must be batched if right tensor is batched"
    end

    if left_batched? and not right_batched? do
      raise ArgumentError, "right tensor must be batched if left tensor is batched"
    end

    # batch axes must be increasing starting from 0
    valid_batch_axes = Enum.to_list(0..(length(b1) - 1))

    # ensure normalized batch axis of left is valid value
    if left_batched? and b1 != valid_batch_axes do
      raise ArgumentError,
            "invalid dot batch axis for the left tensor, batch axes must be successive" <>
              " dimensions starting from 0, got #{inspect(b1)}"
    end

    # ensure normalized batch axis of right is valid value
    if right_batched? and b2 != valid_batch_axes do
      raise ArgumentError,
            "invalid dot batch axis for the right tensor, batch axes must be successive" <>
              " dimensions starting from 0, got #{inspect(b2)}"
    end

    b1_sizes = Enum.map(b1, &elem(s1, &1))
    b2_sizes = Enum.map(b2, &elem(s2, &1))

    # ensure batch dim sizes match if both tensors are batched
    if left_batched? and right_batched? and b1_sizes != b2_sizes do
      raise ArgumentError,
            "dot batch dimension sizes must match, but the left " <>
              "batch dimension of axes #{inspect(b1)} has dimension sizes #{inspect(b1_sizes)}" <>
              "and the right batch dimension of axes #{inspect(b2)} has sizes #{inspect(b2_sizes)}"
    end

    # ensure there is no conflict between left batch axes and left contract axes
    if left_batched? and Enum.any?(b1, &(&1 in c1)) do
      raise ArgumentError,
            "dot batch axes for left tensor (#{inspect(b1)}) cannot be in contract axes" <>
              " (#{inspect(c1)})"
    end

    # ensure there is no conflict between right batch axis and right contract axes
    if right_batched? and Enum.any?(b2, &(&1 in c2)) do
      raise ArgumentError,
            "dot batch axes for right tensor (#{inspect(b2)}) cannot be in contract axes" <>
              " (#{inspect(c2)})"
    end

    :ok
  end

  @doc """
  Returns the shape and names after a Cholesky decomposition.

  ## Examples

      iex> Nx.Shape.cholesky({4, 4}, [:x, :y])
      {{4, 4}, [:x, :y]}

  ## Error Cases

      iex> Nx.Shape.cholesky({3, 2}, [:x, :y])
      ** (ArgumentError) tensor must be a square matrix, got shape: {3, 2}

      iex> Nx.Shape.cholesky({3, 3, 3}, [:x, :y, :z])
      ** (ArgumentError) tensor must have rank 2, got rank 3 with shape {3, 3, 3}
  """
  def cholesky({n, n}, names), do: {{n, n}, names}

  def cholesky({m, n}, _names),
    do: raise(ArgumentError, "tensor must be a square matrix, got shape: #{inspect({m, n})}")

  def cholesky(shape, _names),
    do:
      raise(
        ArgumentError,
        "tensor must have rank 2, got rank #{tuple_size(shape)} with shape #{inspect(shape)}"
      )

  def qr({m, n}, opts) when m >= n do
    mode = opts[:mode]

    case mode do
      :reduced ->
        {{m, n}, {n, n}}

      _ ->
        {{m, m}, {m, n}}
    end
  end

  def qr({m, n}, _opts),
    do:
      raise(
        ArgumentError,
        "tensor must have at least as many rows as columns, got shape: #{inspect({m, n})}"
      )

  def qr(shape, _opts),
    do:
      raise(
        ArgumentError,
        "tensor must have rank 2, got rank #{tuple_size(shape)} with shape #{inspect(shape)}"
      )

  def svd({m, n}) do
    {{m, m}, {n}, {n, n}}
  end

  def svd(shape),
    do:
      raise(
        ArgumentError,
        "tensor must have rank 2, got rank #{tuple_size(shape)} with shape #{inspect(shape)}"
      )

  def lu({n, n}) do
    {{n, n}, {n, n}, {n, n}}
  end

  def lu(shape),
    do:
      raise(
        ArgumentError,
        "tensor must have as many rows as columns, got shape: #{inspect(shape)}"
      )

  def solve({n, n}, {n}), do: :ok
  def solve({n, n}, {n, _m}), do: :ok

  def solve({n, n}, b_shape) do
    raise(
      ArgumentError,
      "`b` tensor has incompatible dimensions, expected #{inspect({n, n})} or {#{n}}, got: " <>
        inspect(b_shape)
    )
  end

  def solve(a_shape, _b_shape) do
    raise(
      ArgumentError,
      "`a` tensor has incompatible dimensions, expected a 2-D tensor with as many rows as columns, got: " <>
        inspect(a_shape)
    )
  end

  defp validate_concat_names!(names) do
    _ =
      Enum.zip_with(names, fn [name | rest] ->
        Enum.reduce(rest, name, &merge_names!(&1, &2))
      end)

    hd(names)
  end

  ## Helpers

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  defp count_down(0, _n), do: []
  defp count_down(i, n), do: [n | count_down(i - 1, n - 1)]

  defp merge_names!(nil, nil), do: nil
  defp merge_names!(nil, name) when is_atom(name), do: name
  defp merge_names!(name, nil) when is_atom(name), do: name
  defp merge_names!(name, name) when is_atom(name), do: name

  defp merge_names!(lhs, rhs),
    do: raise(ArgumentError, "cannot merge names #{inspect(lhs)}, #{inspect(rhs)}")
end
